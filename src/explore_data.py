"""
MEMBER 1 - Dataset exploration (torch ki zaroorat nahi)
Banata hai: results/graphs/class_distribution.png, results/graphs/sample_images.png
Run: python src/explore_data.py [dataset_folder]
"""
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset" / "combined_images"
out = ROOT / "results" / "graphs"
out.mkdir(parents=True, exist_ok=True)
random.seed(42)

classes = sorted(d.name for d in data.iterdir() if d.is_dir())
files = {c: [f for f in (data / c).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}] for c in classes}
counts = {c: len(v) for c, v in files.items()}
print("Class distribution:", counts, "| Total:", sum(counts.values()))

# image size / mode (sample se)
stat = Counter()
for c in classes:
    for f in random.sample(files[c], min(100, len(files[c]))):
        im = Image.open(f)
        stat[(im.size, im.mode)] += 1
print("Image sizes/modes (sample):", dict(stat))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(counts.keys(), counts.values(), color="#5b8def")
ax.bar_label(bars)
ax.set_title("Class distribution"); ax.set_ylabel("Images")
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(out / "class_distribution.png", dpi=150); plt.close()

fig, axes = plt.subplots(len(classes), 4, figsize=(8, 2.1 * len(classes)))
for r, c in enumerate(classes):
    for k, f in enumerate(random.sample(files[c], 4)):
        axes[r, k].imshow(Image.open(f).convert("L"), cmap="gray")
        axes[r, k].axis("off")
        if k == 0:
            axes[r, k].set_title(c, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(out / "sample_images.png", dpi=150); plt.close()
print("Saved graphs in", out)
