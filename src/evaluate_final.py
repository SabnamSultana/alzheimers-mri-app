"""
MEMBER 5: Final held-out test evaluation for CNN and ResNet18.

Expected folders:
  dataset/processed/test/<class_name>/*.png
Expected final checkpoints:
  models/cnn_model.pth
  models/resnet18_model.pth

Run from project root:
  python src/evaluate_final.py

This script NEVER trains a model. It evaluates supplied checkpoints only.
It refuses to substitute Member 4's short benchmark checkpoints.
"""
from pathlib import Path
import json, time, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "dataset" / "processed" / "test"
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"
CM_DIR = RESULTS / "confusion_matrices"
IMG_SIZE, BATCH_SIZE = 224, 32

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)
    )

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final checkpoint: {path}\n"
            "Ask the corresponding teammate to complete training and share this file. "
            "Do not use member4_* benchmark checkpoints as substitutes."
        )
    # Checkpoints are supplied by your project team; load on CPU/GPU as requested.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def extract_state(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key], ckpt
        # Some projects save a bare state_dict.
        if all(isinstance(k, str) for k in ckpt):
            return ckpt, {}
    raise ValueError("Unrecognized checkpoint format.")

def make_model(kind, n_classes):
    if kind == "CNN":
        return CNN(n_classes)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def evaluate(kind, checkpoint_path, loaders, class_names, device):
    ckpt = load_checkpoint(checkpoint_path, device)
    state, metadata = extract_state(ckpt)
    loader = loaders[bool(metadata.get("normalize", False))]
    saved_classes = metadata.get("classes")
    if saved_classes and list(saved_classes) != list(class_names):
        raise ValueError(
            f"{kind} checkpoint class order {saved_classes} does not match "
            f"test folder order {class_names}. Resolve class mapping before evaluation."
        )
    model = make_model(kind, len(class_names)).to(device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = __import__("sklearn.metrics", fromlist=["ConfusionMatrixDisplay"]).ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(f"{kind} - Held-out Test Confusion Matrix")
    plt.tight_layout()
    fig.savefig(CM_DIR / f"{kind.lower()}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Flatten class-wise metrics for a tidy CSV.
    rows = []
    for cls in class_names:
        rows.append({
            "model": kind, "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1_score": report[cls]["f1-score"],
            "support": report[cls]["support"]
        })
    result = {
        "model": kind, "device": device.type, "test_samples": len(y_true),
        "class_names": class_names, "accuracy": acc,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "evaluation_time_seconds": elapsed,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    return result, rows

def main():
    RESULTS.mkdir(exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}. Ensure Member 1's processed "
            "train/val/test dataset is in dataset/processed/."
        )

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
    if len(ds) == 0:
        raise ValueError("Test dataset is empty.")
    class_names = ds.classes
    ds_n = datasets.ImageFolder(TEST_DIR, transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))
    loader = {False: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
              True: DataLoader(ds_n, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Test images:", len(ds), "| Classes:", class_names)

    all_results, all_rows = [], []
    specs = [
        ("CNN", MODEL_DIR / "cnn_model.pth"),
        ("ResNet18", MODEL_DIR / "resnet18_model.pth")
    ]
    for kind, path in specs:
        result, rows = evaluate(kind, path, loader, class_names, device)
        all_results.append(result)
        all_rows.extend(rows)
        print(f"{kind}: accuracy={result['accuracy']:.4f}, "
              f"macro-F1={result['macro_f1']:.4f}")

    with open(RESULTS / "final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(all_rows).to_csv(RESULTS / "classification_report.csv", index=False)

    comparison = [{
        "model": r["model"], "test_samples": r["test_samples"],
        "accuracy": r["accuracy"], "macro_precision": r["macro_precision"],
        "macro_recall": r["macro_recall"], "macro_f1": r["macro_f1"],
        "weighted_f1": r["weighted_f1"],
        "evaluation_time_seconds": r["evaluation_time_seconds"]
    } for r in all_results]
    pd.DataFrame(comparison).to_csv(RESULTS / "model_comparison.csv", index=False)

    comp = pd.DataFrame(comparison).set_index("model")
    ax = comp[["accuracy", "macro_f1", "weighted_f1"]].plot(
        kind="bar", ylim=(0, 1), figsize=(9, 5), rot=0
    )
    ax.set_ylabel("Score")
    ax.set_title("CNN vs ResNet18 - Held-out Test Metrics")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=160)
    plt.close()

    print("\nSaved evaluation files under:", RESULTS)
    print("IMPORTANT: These are test metrics for the supplied final checkpoints.")

if __name__ == "__main__":
    main()
