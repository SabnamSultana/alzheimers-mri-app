"""Near-duplicate leakage audit between train/val/test (catches what SHA-256 cannot).
Run: python src/near_duplicate_audit.py [--thr 0.98]
Writes results/near_duplicate_audit.json"""
import argparse, glob, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def vec(p, s=32):
    a = np.asarray(Image.open(p).convert("L").resize((s, s)), dtype=np.float32).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=0.98)
    a = ap.parse_args()
    files = {s: sorted(glob.glob(str(ROOT / "dataset/processed" / s / "*/*.png"))) for s in ("train", "val", "test")}
    X = {s: np.stack([vec(f) for f in fs]) for s, fs in files.items()}
    out = {"threshold_corr": a.thr, "note": "Pearson correlation of 32x32 grayscale thumbnails vs nearest TRAIN image."}
    for s in ("val", "test"):
        m = (X[s] @ X["train"].T).max(1)
        out[s] = {"n": int(len(m)), "min": float(m.min()), "mean": float(m.mean()), "max": float(m.max()),
                  "near_duplicates_of_train": int((m > a.thr).sum())}
        print(f"{s}: {out[s]['near_duplicates_of_train']}/{len(m)} images are near-duplicates of a train image (corr>{a.thr}); mean={m.mean():.3f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/near_duplicate_audit.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__": main()
