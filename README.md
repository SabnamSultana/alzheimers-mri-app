# GPU-Accelerated Deep Learning-Based Alzheimer's Disease Classification from Brain MRI (Group G-8)

Models: custom CNN and ResNet18 (PyTorch) | Classes: Mild / Moderate / Non / Very Mild Demented
**Academic demonstration only - not a medical device, not clinically validated.**

## Current status (single source of truth)
| Item | Status |
|---|---|
| Code (preprocess, train, evaluate, GPU benchmark) | Complete |
| Final checkpoints | Trained 15 epochs on Tesla T4 (Colab): `models/cnn_model.pth`, `models/resnet18_model.pth` |
| Reported test metrics | CNN 100% / ResNet18 100% on 40 test images |
| **Reliability of those metrics** | **LOW** - see below |
| GPU inference benchmark (final checkpoints) | `results/final_models_gpu_benchmark.*` (T4, forward pass only) |
| Dataset | 240-image subset; source not documented; images appear synthetic |

## Known limitations (read before submitting)
1. **Leakage:** `results/near_duplicate_audit.json` - 36/40 val and 35/40 test images are near-copies (corr > 0.98) of a train image. Exact SHA-256 check (0 duplicates) cannot detect this. The 100% scores therefore measure memorisation, not generalisation.
2. **Tiny test set:** 40 images; one image = 2.5%.
3. **Provenance:** images do not look like real MRI; original 44,000-image dataset not included.
4. `results/final_evaluation.json` was transcribed from Colab output. Run `python src/evaluate_final.py` (needs torch) to regenerate it from the checkpoints.
5. `results/member4_*` are a separate 2-epoch hardware benchmark (training speed-up CNN 8.2x, ResNet18 27.6x, T4 vs CPU). They are **not** classification results.
6. CPU vs GPU inference timings use different setups; do not claim a controlled speed-up from them.
7. Earlier CPU-only run (CNN 3 epochs 72.5%, ResNet18 1 epoch 25%) is superseded by the GPU run and is kept only in `RUN_RESULTS_AND_LIMITATIONS.md` as history.

## How to fix the dataset (recommended before final viva)
```bash
pip install -r requirements.txt
python src/prepare_real_dataset.py --src /path/to/real_dataset --max_per_class 1500   # group-aware split
python src/near_duplicate_audit.py            # should now show ~0 near-duplicates
python src/train_all.py --epochs 15 --batch_size 16 --pretrained   # GPU runtime (Colab T4)
python src/evaluate_final.py
python src/member5_audit.py
python src/gpu_benchmark.py                   # CPU vs GPU training/inference timing
```
`--pretrained` uses ImageNet weights for ResNet18 (usually far better on small medical sets). Select checkpoints with validation only; touch the test set once.

## Prototype demo
`python src/demo_app.py` launches a Gradio web demo (upload MRI -> stage + probabilities + inference time). CLI: `python src/demo_app.py --image x.png --model ResNet18`. Demo video outline: `DEMO_VIDEO_SCRIPT.md`.

## Report
`report/G8_Alzheimers_Assignment_Report.docx` - final assignment report (members, results, CPU-GPU benchmark, limitations). Figures in `results/graphs/` were regenerated from the final 15-epoch run.

## Demo video
`demo_video/G8_Alzheimers_Project_Walkthrough.mp4` - 3.5 min narrated walkthrough (clear neural English voice) (slides with real results). It is a slide-based walkthrough, not a screen recording of the live app; narration text in `demo_video/NARRATION_SCRIPT.md`. Record a live screen demo with `DEMO_VIDEO_SCRIPT.md` if the course needs one.

## Layout
`src/` code | `models/` checkpoints | `results/` metrics, graphs, benchmarks | `report/` report sections | `notebooks/` CNN notebook | `dataset/` data + README

## Team
M1 data prep | M2 CNN | M3 ResNet18 | M4 CPU-GPU benchmark | M5 audit, evaluation, report

Note: the four `member4_*.pth` benchmark checkpoints (~90 MB) were removed to keep this zip small. Their results remain in `results/member4_*`.
