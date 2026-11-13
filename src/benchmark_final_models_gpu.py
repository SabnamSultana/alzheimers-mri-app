"""Benchmark the project's *trained final checkpoints* on CUDA.

Run from the extracted project root:
  python src/benchmark_final_models_gpu.py
Optional:
  python src/benchmark_final_models_gpu.py --batch-sizes 1 8 16 32 --repeats 100

This script does NOT retrain models and does NOT use Member 4's random-init
benchmark checkpoints. It loads models/cnn_model.pth and
models/resnet18_model.pth, evaluates the held-out test folder, and measures
CUDA inference latency, throughput, and peak allocated GPU memory.
"""
from pathlib import Path
import argparse, csv, json, platform, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'dataset' / 'processed' / 'test'
MODEL_DIR = ROOT / 'models'
RESULTS = ROOT / 'results'


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))

class CNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.features = nn.Sequential(conv_block(3,32), conv_block(32,64),
                                      conv_block(64,128), conv_block(128,256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.5), nn.Linear(256,128),
                                        nn.ReLU(inplace=True), nn.Dropout(.3), nn.Linear(128,n))
    def forward(self, x): return self.classifier(self.pool(self.features(x)))

def make_model(kind, n):
    if kind == 'CNN': return CNN(n)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m

def state_from_checkpoint(path):
    if not path.exists(): raise FileNotFoundError(f'Missing trained checkpoint: {path}')
    try: ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError: ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for k in ('model_state', 'model_state_dict', 'state_dict'):
            if k in ckpt: return ckpt[k], ckpt
        if all(isinstance(k, str) for k in ckpt): return ckpt, {}
    raise ValueError(f'Unsupported checkpoint format: {path}')

def sync(): torch.cuda.synchronize()

@torch.inference_mode()
def benchmark(model, loader, device, repeats, warmup, batch_size):
    model.eval()
    # Warm up using the first actual test batch; CUDA kernels/context initialize here.
    first, _ = next(iter(loader))
    first = first.to(device, non_blocking=True)
    for _ in range(warmup): model(first)
    sync()

    # End-to-end model-forward benchmark: input batches already reside on GPU.
    # This isolates model inference from disk and host-to-device transfer.
    gpu_batches = []
    for x, _ in loader:
        gpu_batches.append(x.to(device, non_blocking=True))
    sync()
    torch.cuda.reset_peak_memory_stats(device)
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_images = 0
    start_evt.record()
    for _ in range(repeats):
        for x in gpu_batches:
            model(x)
            total_images += x.size(0)
    end_evt.record(); sync()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    return {
        'batch_size_setting': batch_size,
        'warmup_batches': warmup,
        'repeats_over_full_test_set': repeats,
        'images_processed': total_images,
        'total_inference_seconds': elapsed_ms / 1000,
        'latency_ms_per_image': elapsed_ms / total_images,
        'throughput_images_per_second': total_images / (elapsed_ms / 1000),
        'peak_allocated_gpu_memory_mb': torch.cuda.max_memory_allocated(device) / (1024**2),
        'peak_reserved_gpu_memory_mb': torch.cuda.max_memory_reserved(device) / (1024**2),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 16, 32])
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--warmup', type=int, default=10)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU, then rerun.')
    if not TEST_DIR.exists(): raise FileNotFoundError(f'Test images not found: {TEST_DIR}')

    device = torch.device('cuda:0')
    # Matches evaluate_final.py preprocessing: resize + ToTensor, no added normalization.
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
    classes = ds.classes
    all_results = {
        'benchmark_type': 'trained_final_checkpoint_cuda_inference',
        'hardware': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'python_version': platform.python_version(),
        'cuda_version': torch.version.cuda,
        'test_samples': len(ds), 'classes': classes,
        'input_size': [3,224,224], 'timing_method': 'CUDA events with synchronization',
        'note': 'Forward-pass only; test batches are preloaded to GPU, excluding disk and host-to-device transfer.',
        'models': {}
    }
    rows = []
    for kind, ckpt_path in [('CNN', MODEL_DIR/'cnn_model.pth'), ('ResNet18', MODEL_DIR/'resnet18_model.pth')]:
        state, meta = state_from_checkpoint(ckpt_path)
        saved_classes = meta.get('classes') if isinstance(meta, dict) else None
        if saved_classes and list(saved_classes) != list(classes):
            raise ValueError(f'{kind} checkpoint class order differs from test ImageFolder order: {saved_classes} vs {classes}')
        model = make_model(kind, len(classes)).to(device)
        model.load_state_dict(state)
        model.eval()
        model_results = []
        for bs in args.batch_sizes:
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
            torch.cuda.empty_cache()
            r = benchmark(model, loader, device, args.repeats, args.warmup, bs)
            model_results.append(r)
            rows.append({'model':kind, 'gpu':all_results['hardware'], **r})
            print(f"{kind} | batch={bs}: {r['latency_ms_per_image']:.4f} ms/image, "
                  f"{r['throughput_images_per_second']:.2f} images/s, "
                  f"peak allocated={r['peak_allocated_gpu_memory_mb']:.1f} MB")
        all_results['models'][kind] = model_results
        del model
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'final_models_gpu_benchmark.json').write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    with (RESULTS/'final_models_gpu_benchmark.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('\nSaved results:')
    print(RESULTS/'final_models_gpu_benchmark.json')
    print(RESULTS/'final_models_gpu_benchmark.csv')

if __name__ == '__main__': main()
