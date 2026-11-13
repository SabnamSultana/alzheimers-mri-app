"""
MEMBER 3 - ResNet18 training for G-8 Alzheimer's MRI classification.

This script is designed to work with Member 1 + Member 2's combined project:
- Uses src/data_preprocessing.py and its shared results/split.csv
- Uses the same train/validation/test split as the CNN
- Saves ResNet18 weights, graphs, and results

Run from the project root in Colab:
    !python src/resnet18_train.py --epochs 1
Then, for a longer run:
    !python src/resnet18_train.py --epochs 15

Expected dataset folder:
    dataset/combined_images/
        MildDemented/
        ModerateDemented/
        NonDemented/
        VeryMildDemented/

IMPORTANT:
The ZIP shared by Members 1 and 2 contains the code and example graphs, but not the actual MRI image files.
Ask Member 1 to provide the dataset and put it in dataset/combined_images/.
"""

import argparse
import copy
import json
import os
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure imports work when run as: python src/resnet18_train.py
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
from data_preprocessing import get_dataloaders, SEED, ROOT as DATA_ROOT


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, n = 0.0, 0, 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return loss_sum / max(n, 1), correct / max(n, 1), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(DATA_ROOT / "dataset" / "combined_images"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(DATA_ROOT / "models" / "resnet18_model.pth"))
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights (may need internet on first run)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "The shared ZIP has the Member 1/2 code but no MRI images. "
            "Ask Member 1 for the dataset, extract it to dataset/combined_images, "
            "then run this script again."
        )

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        csv_path=DATA_ROOT / "results" / "split.csv",
    )
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    weights = None
    if args.pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_time_s": []
    }
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    os.makedirs(Path(args.out).parent, exist_ok=True)
    graph_dir = DATA_ROOT / "results" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_time = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, _ = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(train_time)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"train time {train_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state": best_state,
                "classes": class_names,
                "img_size": args.img_size,
                "architecture": "ResNet18",
            }, args.out)

    total_time = time.perf_counter() - start_total
    model.load_state_dict(best_state)
    # Detailed test metrics, computed only after model selection is complete.
    model.eval()
    y_true, y_pred = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_start = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_time = time.perf_counter() - test_start
    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax_cm, xticks_rotation=45, colorbar=False)
    ax_cm.set_title("ResNet18 Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(graph_dir / "resnet18_confusion_matrix.png", dpi=150)
    plt.close(fig_cm)

    print(
        f"\nTotal training time: {total_time:.1f}s"
        f" | Best validation accuracy: {best_val_acc:.4f}"
        f" | Test accuracy: {test_acc:.4f}"
        f" | Test evaluation time: {test_time:.1f}s"
    )

    epochs_axis = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_axis, history["train_loss"], label="Train")
    axes[0].plot(epochs_axis, history["val_loss"], label="Validation")
    axes[0].set_title("ResNet18 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_axis, history["train_acc"], label="Train")
    axes[1].plot(epochs_axis, history["val_acc"], label="Validation")
    axes[1].set_title("ResNet18 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(graph_dir / "resnet18_training_curves.png", dpi=150)
    plt.close(fig)

    result = {
        "model": "ResNet18",
        "device": device.type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "class_names": class_names,
        "total_train_time_s": total_time,
        "avg_epoch_train_time_s": sum(history["epoch_time_s"]) / len(history["epoch_time_s"]),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_evaluation_time_s": test_time,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    results_path = DATA_ROOT / "results" / f"resnet18_results_{device.type}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSaved model:", args.out)
    print("Saved graphs:", graph_dir / "resnet18_training_curves.png", "and resnet18_confusion_matrix.png")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Saved metrics:", results_path)


if __name__ == "__main__":
    main()
