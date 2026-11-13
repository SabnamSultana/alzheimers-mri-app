"""
ALL-MEMBER TRAINING PIPELINE
Trains the agreed CNN and ResNet18 on the SAME pre-split data.
Does not use the test set for model selection.
Run from project root:
  python src/train_all.py --epochs 15
Expected:
  dataset/processed/train/<class>/
  dataset/processed/val/<class>/
  dataset/processed/test/<class>/
"""
import argparse, json, time, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cnn_model import CNN

def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    if device.type == "cuda": torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    if device.type == "cuda": torch.cuda.synchronize()
    return total_loss/max(total,1), correct/max(total,1), time.perf_counter()-start

def build_model(name, n, pretrained=False):
    if name == "CNN": return CNN(n)
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT/"dataset"/"processed"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet-pretrained ResNet18 (+ImageNet normalisation)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable. Select a CUDA runtime or use --device cpu.")
    data = Path(args.data_dir)
    required = [data/"train", data/"val", data/"test"]
    if not all(p.exists() for p in required):
        raise FileNotFoundError(f"Expected train/val/test folders under {data}")
    ops = [transforms.Resize((224,224)), transforms.ToTensor()]
    if args.pretrained: ops.append(transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]))
    tfm = transforms.Compose(ops)
    train_ds = datasets.ImageFolder(data/"train", transform=tfm)
    val_ds = datasets.ImageFolder(data/"val", transform=tfm)
    test_ds = datasets.ImageFolder(data/"test", transform=tfm)
    classes = train_ds.classes
    for name, ds in [("validation",val_ds),("test",test_ds)]:
        if ds.classes != classes: raise ValueError(f"{name} class order differs from training.")
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise ValueError("One or more dataset splits are empty.")
    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_dl = DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_dl = DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    (ROOT/"models").mkdir(exist_ok=True)
    (ROOT/"results"/"graphs").mkdir(parents=True,exist_ok=True)
    print(f"Device={device}; classes={classes}; sizes={len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    for name in ("CNN","ResNet18"):
        model = build_model(name,len(classes),args.pretrained and name=="ResNet18").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr if name=="CNN" else min(args.lr,1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=2,factor=0.5)
        best_val_loss=float("inf"); history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
        checkpoint = ROOT/"models"/("cnn_model.pth" if name=="CNN" else "resnet18_model.pth")
        start_all=time.perf_counter()
        for ep in range(1,args.epochs+1):
            trl,tra,trt=epoch_pass(model,train_dl,criterion,device,optimizer)
            vl,va,vt=epoch_pass(model,val_dl,criterion,device)
            scheduler.step(vl)
            for k,v in zip(history,(trl,tra,vl,va)): history[k].append(v)
            if vl < best_val_loss:
                best_val_loss=vl
                torch.save({"model_state":model.state_dict(),"classes":classes,
                            "model_name":name,"epoch":ep,"seed":args.seed,
                            "image_size":224,"normalize":bool(args.pretrained),"pretrained":bool(args.pretrained and name=="ResNet18"),"validation_loss":vl},checkpoint)
            print(f"{name} epoch {ep}/{args.epochs}: train_acc={tra:.4f}, val_acc={va:.4f}, val_loss={vl:.4f}")
        total_train=time.perf_counter()-start_all
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].plot(history["train_loss"],label="Train"); ax[0].plot(history["val_loss"],label="Validation")
        ax[0].set_title(f"{name} loss"); ax[0].legend()
        ax[1].plot(history["train_acc"],label="Train"); ax[1].plot(history["val_acc"],label="Validation")
        ax[1].set_title(f"{name} accuracy"); ax[1].legend()
        plt.tight_layout(); plt.savefig(ROOT/"results"/"graphs"/f"{name.lower()}_training_curves.png",dpi=150); plt.close()
        with open(ROOT/"results"/f"{name.lower()}_training_history.json","w") as f:
            json.dump({"model":name,"device":device.type,"epochs":args.epochs,"total_training_seconds":total_train,
                       "classes":classes,"history":history,"checkpoint":str(checkpoint.relative_to(ROOT))},f,indent=2)
        print(f"Saved best-validation checkpoint: {checkpoint}")
    print("\nTraining complete. Next run: python src/evaluate_final.py")
    print("Test data was not used for model selection.")

if __name__=="__main__": main()
