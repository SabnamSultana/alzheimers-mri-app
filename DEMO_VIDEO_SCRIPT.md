# Demo video guide (2-4 min, screen recording on Colab T4)
1. (15 s) Title: project name, Group 8, members, goal: classify brain MRI into 4 stages, CNN vs ResNet18, GPU-accelerated.
2. (20 s) Show folder structure: `src/`, `models/`, `results/`, `dataset/`.
3. (30 s) Colab: `!nvidia-smi` (show Tesla T4) and `torch.cuda.is_available()` -> True.
4. (40 s) Run `python src/train_all.py --epochs 15` (or show the saved training curves in `results/`), mention CPU vs GPU training speed-up from `results/member4_*` (CNN 8.2x, ResNet18 27.6x).
5. (40 s) Run `python src/demo_app.py`, upload 2-3 images from `dataset/sample_mri/`, show predicted stage, probabilities and inference ms (GPU).
6. (30 s) Show `results/` confusion matrices, accuracy table and GPU inference benchmark.
7. (20 s) Be honest: small subset, near-duplicate leakage (`near_duplicate_audit.json`), next step = real dataset + pretrained ResNet18.
