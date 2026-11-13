"""
MEMBER 4 - Reproducible CPU/GPU performance benchmark for CNN and ResNet18.

Run from the project root:
  python src/gpu_benchmark.py --device cpu --epochs 1
  python src/gpu_benchmark.py --device cuda --epochs 1
  python src/gpu_benchmark.py --compare

Use identical epochs, batch size, image size, and dataset split for both runs.
The script benchmarks both models with random initialization (no pretrained download),
records training time, inference latency/throughput, peak device memory and hardware.
"""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cnn_model import CNN
from data_preprocessing import get_dataloaders, SEED


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=0):
    model.train()
    seen, loss_total = 0, 0.0
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * images.size(0)
        seen += images.size(0)
    sync(device)
    elapsed = time.perf_counter() - start
    return {"seconds": elapsed, "samples": seen,
            "samples_per_second": seen / elapsed if elapsed else None,
            "mean_loss": loss_total / seen if seen else None}


@torch.no_grad()
def benchmark_inference(model, loader, device, repeats=30):
    model.eval()
    try:
        images, _ = next(iter(loader))
    except StopIteration:
        raise RuntimeError("Test loader is empty. Check the dataset and split.")
    images = images.to(device)
    # Warm-up avoids including one-time backend initialization in measurements.
    for _ in range(5):
        model(images)
    sync(device)
    start = time.perf_counter()
    total_images = 0
    for _ in range(repeats):
        model(images)
        total_images += images.size(0)
    sync(device)
    elapsed = time.perf_counter() - start
    return {
        "batch_size": int(images.size(0)),
        "repeats": repeats,
        "total_seconds": elapsed,
        "latency_ms_per_image": elapsed * 1000 / total_images,
        "throughput_images_per_second": total_images / elapsed if elapsed else None,
    }


def make_model(name, nclasses, device):
    if name == "CNN":
        return CNN(nclasses).to(device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, nclasses)
    return model.to(device)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU.")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}. The received ZIP contains code "
            "but no MRI images. Ask Member 1 for dataset/combined_images/."
        )
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=data_dir, batch_size=args.batch_size, img_size=args.img_size,
        csv_path=ROOT / "results" / "split.csv"
    )
    if args.max_batches:
        print(f"SMOKE/QUICK MODE: maximum {args.max_batches} training batches per epoch.")
    results = {
        "device": device.type,
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.platform(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.img_size,
        "max_batches_per_epoch": args.max_batches or "all",
        "seed": SEED,
        "models": {}
    }

    for name in ("CNN", "ResNet18"):
        print(f"\n=== {name} | {device} ===")
        model = make_model(name, len(classes), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_stats = []
        total_start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            stats = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, args.max_batches)
            epoch_stats.append(stats)
            print(f"Epoch {epoch}/{args.epochs}: {stats['seconds']:.2f}s, "
                  f"{stats['samples_per_second']:.2f} images/s")
        sync(device)
        total_training = time.perf_counter() - total_start
        inference = benchmark_inference(model, test_loader, device, args.inference_repeats)
        peak_mem = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else None)
        results["models"][name] = {
            "total_training_seconds": total_training,
            "average_epoch_seconds": sum(x["seconds"] for x in epoch_stats) / len(epoch_stats),
            "epoch_details": epoch_stats,
            "inference": inference,
            "peak_cuda_memory_mb": peak_mem,
        }
        print(f"Total training: {total_training:.2f}s | "
              f"Inference: {inference['latency_ms_per_image']:.3f} ms/image | "
              f"Throughput: {inference['throughput_images_per_second']:.2f} images/s")
        if peak_mem is not None:
            print(f"Peak allocated CUDA memory: {peak_mem:.1f} MB")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"member4_benchmark_{device.type}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark: {out_path}")
    print("Note: this benchmark trains each model from a fresh random initialization; "
          "it is a hardware-performance experiment, not a final accuracy evaluation.")


def compare():
    result_dir = ROOT / "results"
    paths = [result_dir / "member4_benchmark_cpu.json",
             result_dir / "member4_benchmark_cuda.json"]
    records = []
    for path in paths:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for model, metrics in data["models"].items():
                records.append({
                    "device": data["device"], "hardware": data["hardware"],
                    "model": model, "epochs": data["epochs"],
                    "batch_size": data["batch_size"],
                    "training_seconds": round(metrics["total_training_seconds"], 3),
                    "avg_epoch_seconds": round(metrics["average_epoch_seconds"], 3),
                    "inference_ms_per_image": round(metrics["inference"]["latency_ms_per_image"], 4),
                    "inference_images_per_second": round(metrics["inference"]["throughput_images_per_second"], 3),
                    "peak_cuda_memory_mb": metrics["peak_cuda_memory_mb"],
                })
    if not records:
        raise FileNotFoundError("No benchmark JSON found. Run CPU and/or CUDA benchmark first.")
    csv_path = result_dir / "member4_performance_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Plot training time by model/device, with labels sourced from actual result files.
    labels = [f"{r['model']}\\n{r['device'].upper()}" for r in records]
    values = [r["training_seconds"] for r in records]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.bar_label(bars, fmt="%.1f s")
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("CNN and ResNet18: CPU/GPU Training-Time Comparison")
    plt.tight_layout()
    graph = result_dir / "graphs" / "member4_training_time_comparison.png"
    graph.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(graph, dpi=160)
    plt.close(fig)

    # Separate inference chart.
    vals = [r["inference_ms_per_image"] for r in records]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals)
    ax.bar_label(bars, fmt="%.3f ms")
    ax.set_ylabel("Inference latency (ms/image)")
    ax.set_title("CNN and ResNet18: CPU/GPU Inference-Latency Comparison")
    plt.tight_layout()
    graph2 = result_dir / "graphs" / "member4_inference_comparison.png"
    plt.savefig(graph2, dpi=160)
    plt.close(fig)
    print("Saved:", csv_path)
    print("Saved:", graph)
    print("Saved:", graph2)
    print("Rows:", len(records))
    for r in records:
        print(r)


def main():
    p = argparse.ArgumentParser(description="Member 4 CPU/GPU benchmark")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--data_dir", default=str(ROOT / "dataset" / "combined_images"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_batches", type=int, default=0,
                   help="Optional quick smoke test; 0 means full training split")
    p.add_argument("--inference_repeats", type=int, default=30)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        compare()
    else:
        run(args)


if __name__ == "__main__":
    main()
