"""
MEMBER 2 - CNN training (Member 1 ke data loaders se connected)
Run:  python src/train.py --data_dir dataset/combined_images --epochs 15
Member 4 CPU vs GPU ke liye:  --device cpu   /   --device cuda
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from cnn_model import CNN
from data_preprocessing import get_dataloaders, SEED, ROOT


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(ROOT / "dataset" / "combined_images"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "models" / "cnn_model.pth"))
    a = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device(a.device)
    train_dl, val_dl, test_dl, classes = get_dataloaders(a.data_dir, a.batch_size, a.img_size)
    print("Classes:", classes, "| Device:", device)

    model = CNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, epoch_times = 0.0, []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(ROOT / "results" / "graphs", exist_ok=True)

    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        scheduler.step(vl)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if va > best_val:
            best_val = va
            torch.save({"model_state": model.state_dict(), "classes": classes}, a.out)
        print(f"Epoch {ep:02d}/{a.epochs} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {epoch_times[-1]:.1f}s")
    total_time = time.time() - t_start

    # best model se test accuracy
    model.load_state_dict(torch.load(a.out, map_location=device)["model_state"])
    _, test_acc = run_epoch(model, test_dl, criterion, device)
    print(f"\nTotal training time: {total_time:.1f}s | Best val acc: {best_val:.4f} | Test acc: {test_acc:.4f}")

    ep_range = range(1, a.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep_range, hist["train_loss"], label="Train"); ax[0].plot(ep_range, hist["val_loss"], label="Val")
    ax[0].set_title("CNN Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep_range, hist["train_acc"], label="Train"); ax[1].plot(ep_range, hist["val_acc"], label="Val")
    ax[1].set_title("CNN Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(ROOT / "results" / "graphs" / "cnn_training_curves.png", dpi=150)

    with open(ROOT / "results" / f"cnn_results_{device.type}.json", "w") as f:
        json.dump({"device": device.type, "epochs": a.epochs, "batch_size": a.batch_size,
                   "total_train_time_s": total_time, "avg_epoch_time_s": sum(epoch_times) / len(epoch_times),
                   "best_val_acc": best_val, "test_acc": test_acc, "history": hist}, f, indent=2)


if __name__ == "__main__":
    main()
