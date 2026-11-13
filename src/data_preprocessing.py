"""
MEMBER 1 - Dataset preparation
- images scan karta hai, duplicate hataata hai, stratified train/val/test split banata hai
- split ko results/split.csv me save karta hai (Member 2, 3, 5 sab SAME split use karenge)
- PyTorch DataLoaders return karta hai
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent   # project root
SEED = 42
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet (ResNet18 ke liye bhi sahi)
EXTS = {".jpg", ".jpeg", ".png"}


def build_split_csv(data_dir, csv_path=ROOT / "results" / "split.csv", val=0.15, test=0.15):
    """Folder scan -> duplicate remove -> stratified split -> CSV. Agar CSV already hai to reuse hota hai."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    data_dir = Path(data_dir)
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    rows, seen = [], set()
    for label, cls in enumerate(classes):
        for f in sorted((data_dir / cls).iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen:          # exact duplicate -> skip
                continue
            seen.add(h)
            rows.append((str(f), cls, label))
    df = pd.DataFrame(rows, columns=["path", "class_name", "label"])

    train_df, temp_df = train_test_split(
        df, test_size=val + test, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=test / (val + test), stratify=temp_df["label"], random_state=SEED)
    out = pd.concat([train_df.assign(split="train"), val_df.assign(split="val"),
                     test_df.assign(split="test")]).reset_index(drop=True)

    os.makedirs(os.path.dirname(str(csv_path)), exist_ok=True)
    out.to_csv(csv_path, index=False)
    return out


class MRIDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")   # grayscale + RGB dono ko 3-channel bana deta hai
        return self.transform(img), self.labels[i]


def get_transforms(img_size=IMG_SIZE, augment=False):
    ops = [transforms.Resize((img_size, img_size))]
    if augment:
        ops += [transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def get_dataloaders(data_dir, batch_size=32, img_size=IMG_SIZE, num_workers=2,
                    csv_path=ROOT / "results" / "split.csv", augment_train=True):
    df = build_split_csv(data_dir, csv_path)
    class_names = df.sort_values("label")["class_name"].unique().tolist()
    pin = torch.cuda.is_available()

    def make(split, tf, shuffle):
        d = df[df["split"] == split]
        return DataLoader(MRIDataset(d, tf), batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin)

    train = make("train", get_transforms(img_size, augment_train), True)
    val = make("val", get_transforms(img_size), False)
    test = make("test", get_transforms(img_size), False)
    return train, val, test, class_names


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "dataset" / "combined_images")
    df = build_split_csv(d)
    print("Total images (duplicates hatane ke baad):", len(df))
    print(df.groupby(["split", "class_name"]).size().unstack(0))
