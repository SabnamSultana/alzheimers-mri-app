"""Build a leakage-safe train/val/test split from a REAL MRI dataset.
Near-duplicate images (and augmented copies) are clustered and each cluster goes to ONE split only.
Usage:
  python src/prepare_real_dataset.py --src /path/to/dataset --max_per_class 1500
where --src contains one sub-folder per class (e.g. MildDemented/, ModerateDemented/, NonDemented/, VeryMildDemented/).
Output: dataset/processed/{train,val,test}/<class>/  (80/10/10, group-aware)."""
import argparse, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
EXT = {".png", ".jpg", ".jpeg"}

def vec(p, s=16):
    a = np.asarray(Image.open(p).convert("L").resize((s, s)), dtype=np.float32).ravel()
    a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)

def clusters(X, thr):
    n = len(X); parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for s in range(0, n, 1000):
        sim = X[s:s+1000] @ X.T
        for i, j in zip(*np.where(sim > thr)):
            i += s
            if i != j: parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--thr", type=float, default=0.985)
    ap.add_argument("--max_per_class", type=int, default=1500); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = np.random.RandomState(a.seed)
    classes = sorted(d.name for d in Path(a.src).iterdir() if d.is_dir())
    paths, labels = [], []
    for k, c in enumerate(classes):
        fs = sorted(f for f in (Path(a.src) / c).iterdir() if f.suffix.lower() in EXT)
        if len(fs) > a.max_per_class: fs = [fs[i] for i in rng.choice(len(fs), a.max_per_class, replace=False)]
        paths += fs; labels += [k] * len(fs)
    print(f"{len(paths)} images, classes={classes}")
    X = np.stack([vec(p) for p in paths]); labels = np.array(labels)
    groups = clusters(X, a.thr)
    print(f"{len(set(groups))} near-duplicate clusters (thr={a.thr}); each cluster stays in one split")
    if len(set(groups)) < 50:
        raise SystemExit(f"Only {len(set(groups))} distinct clusters: images are almost all near-identical, so no leakage-free split is possible. Use a real MRI dataset (not synthetic/augmented copies).")
    folds = list(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=a.seed).split(X, labels, groups))
    split = np.array(["train"] * len(paths), dtype=object)
    split[folds[0][1]] = "test"; split[folds[1][1]] = "val"
    out = ROOT / "dataset/processed"
    if out.exists(): shutil.rmtree(out)
    for p, y, s in zip(paths, labels, split):
        d = out / s / classes[y]; d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / f"{classes[y]}_{p.stem}{p.suffix}")
    for s in ("train", "val", "test"): print(s, int((split == s).sum()))
    print("Next: python src/near_duplicate_audit.py ; python src/train_all.py --pretrained")

if __name__ == "__main__": main()
