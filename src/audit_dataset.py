"""Audit split sizes, class consistency, and cross-split exact duplicates."""
from pathlib import Path
from collections import defaultdict
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset"/"processed"
splits=("train","val","test")
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
classes_by_split={}
hashes=defaultdict(list)
summary={}
for split in splits:
    folder=BASE/split
    if not folder.exists(): raise FileNotFoundError(folder)
    classes=sorted(p.name for p in folder.iterdir() if p.is_dir())
    classes_by_split[split]=classes
    summary[split]={}
    for cls in classes:
        imgs=[p for p in (folder/cls).rglob("*") if p.is_file()]
        summary[split][cls]=len(imgs)
        for p in imgs: hashes[sha(p)].append((split,cls,str(p.relative_to(ROOT))))
if len({tuple(v) for v in classes_by_split.values()}) != 1:
    raise ValueError(f"Class folders differ across splits: {classes_by_split}")
duplicates={k:v for k,v in hashes.items() if len({x[0] for x in v})>1}
report={"counts":summary,"total_images":sum(sum(x.values()) for x in summary.values()),
        "classes":classes_by_split["train"],"exact_duplicate_hashes_across_splits":len(duplicates),
        "duplicate_examples":list(duplicates.values())[:20]}
out=ROOT/"results"/"data_integrity_audit.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print("\nNote: exact file hashes do not detect resized/recompressed copies or patient-level leakage.")
