"""Download a REAL labelled 4-class Alzheimer MRI dataset (run in Colab / any machine with internet).
Source: Hugging Face 'Falah/Alzheimer_MRI' (Apache-2.0; derived from the public Kaggle Alzheimer MRI dataset, ~6,400 images,
classes: Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented).
Usage:
  pip install datasets pillow
  python src/download_real_dataset.py --out raw_alzheimer
  python src/prepare_real_dataset.py --src raw_alzheimer --max_per_class 1500   # leakage-safe split
  python src/near_duplicate_audit.py
  python src/train_all.py --epochs 15 --pretrained
"""
import argparse
from pathlib import Path
from datasets import load_dataset

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="raw_alzheimer"); a = ap.parse_args()
out = Path(a.out); n = 0
for split in ("train", "test"):
    ds = load_dataset("Falah/Alzheimer_MRI", split=split)
    names = ds.features["label"].names if hasattr(ds.features["label"], "names") else ["Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented"]
    for i, ex in enumerate(ds):
        d = out / names[ex["label"]].replace(" ", "_"); d.mkdir(parents=True, exist_ok=True)
        ex["image"].convert("RGB").save(d / f"{split}_{i:05d}.png"); n += 1
print(f"saved {n} images to {out}/ (class sub-folders)")
