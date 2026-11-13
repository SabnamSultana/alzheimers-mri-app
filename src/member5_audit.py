"""Validate the executed project deliverables and report remaining submission caveats."""
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
required = {
    "Processed training split": ROOT / "dataset/processed/train",
    "Processed validation split": ROOT / "dataset/processed/val",
    "Processed test split": ROOT / "dataset/processed/test",
    "Final CNN checkpoint": ROOT / "models/cnn_model.pth",
    "Final ResNet18 checkpoint": ROOT / "models/resnet18_model.pth",
    "Dataset integrity audit": ROOT / "results/data_integrity_audit.json",
    "Final evaluation JSON": ROOT / "results/final_evaluation.json",
    "Classification report CSV": ROOT / "results/classification_report.csv",
    "Model comparison CSV": ROOT / "results/model_comparison.csv",
    "CNN confusion matrix": ROOT / "results/confusion_matrices/cnn_confusion_matrix.png",
    "ResNet18 confusion matrix": ROOT / "results/confusion_matrices/resnet18_confusion_matrix.png",
    "CNN training history": ROOT / "results/cnn_training_history.json",
    "ResNet18 training history": ROOT / "results/resnet18_training_history.json",
    "CPU inference benchmark": ROOT / "results/final_cpu_inference_benchmark.json",
}
print("FINAL PROJECT DELIVERABLE AUDIT")
missing = []
for label, path in required.items():
    ok = path.exists()
    print(f"{'FOUND  ' if ok else 'MISSING'} | {label}: {path.relative_to(ROOT)}")
    if not ok:
        missing.append(label)

for name in ("cnn_training_history.json", "resnet18_training_history.json"):
    p = ROOT / "results" / name
    if p.exists():
        d = json.loads(p.read_text())
        print(f"INFO   | {d['model']} epochs completed: {d['epochs']} on {d['device']}")

eval_path = ROOT / "results/final_evaluation.json"
if eval_path.exists():
    d = json.loads(eval_path.read_text())
    items = d if isinstance(d, list) else [dict(model=k, test_samples=d.get("test_samples"), **v) for k, v in d["models"].items()]
    for r in items:
        print(f"METRIC | {r['model']}: accuracy={r['accuracy']:.4f}, macro-F1={r['macro_f1']:.4f}, n={r['test_samples']}")

audit_path = ROOT / "results/data_integrity_audit.json"
if audit_path.exists():
    a = json.loads(audit_path.read_text())
    print(f"AUDIT  | total_images={a['total_images']}, exact_cross_split_duplicate_hashes={a['exact_duplicate_hashes_across_splits']}")

print("\nLIMITATION | Included results come from a 240-image subset (40 test images); perfect scores are NOT reliable.")
nd = ROOT / "results/near_duplicate_audit.json"
if nd.exists():
    t = json.loads(nd.read_text())["test"]
    print(f"LIMITATION | Near-duplicate audit: {t['near_duplicates_of_train']}/{t['n']} test images are near-copies of train images (leakage).")
print("LIMITATION | Re-train on a real, leakage-safe split: see README section 'How to fix the dataset'.")
sys.exit(1 if missing else 0)
