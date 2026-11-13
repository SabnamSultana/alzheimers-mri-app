# GPU-Accelerated Deep Learning-Based Alzheimer's Disease Classification from Brain MRI Images

## Project status and evidence boundary
This report covers a four-class Alzheimer MRI classification project (custom CNN and ResNet18) with CPU/GPU benchmarking. Both final models were trained for 15 epochs on an NVIDIA Tesla T4 (Google Colab) using a 240-image processed subset (160 train, 40 validation, 40 test). The original ~44,000-image collection is not included. Reported test metrics are **preliminary and unreliable** (see Limitations).

## Abstract
This project implements a four-class brain MRI image classification workflow using a custom convolutional neural network (CNN) and ResNet18. The workflow covers dataset inspection and preprocessing, model training, CPU/GPU benchmarking, held-out test evaluation, and comparative reporting. The supplied hardware benchmark records faster execution on an NVIDIA T4 CUDA GPU than on CPU for the measured two-epoch runs. These are hardware benchmark results, not evidence of final diagnostic classification performance.

## Problem statement
The project investigates whether deep learning models can classify supplied brain MRI images into the dataset's four folder-defined categories: Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented. This is an academic image-classification task; it is not a clinically validated diagnostic system.

## Objectives
1. Organize and inspect the MRI image dataset.
2. Apply consistent image resizing and tensor conversion.
3. Train a custom CNN and ResNet18 using the same train/validation/test split.
4. Measure CPU/GPU training and inference performance.
5. Evaluate final checkpoints on the held-out test split and audit the split for leakage.
6. Compare accuracy, macro precision, macro recall, macro F1, weighted F1, and evaluation time.

## Dataset and preprocessing
The supplied project README describes a four-class combined image collection of approximately 44,000 MRI slices and notes that the collection may be augmented/oversampled. The complete original collection was not included in the handoff archive. The archive's processed subset contains:
- Training: 160 images (40 per class)
- Validation: 40 images (10 per class)
- Testing: 40 images (10 per class)
- Total processed subset: 240 images

Folder labels are:
- Mild Demented
- Moderate Demented
- Non Demented
- Very Mild Demented

Images are resized to 224 × 224 and converted to tensors. The training pipeline preserves the folder-derived class order across all splits and checks that the class folders match. A split-level exact SHA-256 duplicate audit is provided. This audit cannot detect transformed duplicates or patient-level leakage. If patient identifiers are available, a patient-level split is preferable.

## Model 1: Custom CNN
The supplied CNN uses four convolutional blocks with channel widths 32, 64, 128, and 256. Each block contains convolution, batch normalization, ReLU, and max pooling. Adaptive average pooling feeds a classifier with dropout and fully connected layers. The final layer outputs four class scores.

## Model 2: ResNet18
The project uses the ResNet18 architecture from Torchvision and replaces its final fully connected layer with a four-class output layer. The supplied training code supports training from random initialization and optionally supports ImageNet pretrained weights. The unified training script defaults to random initialization so it can run without downloading pretrained weights.

## Training methodology
The unified script `src/train_all.py` trains both models on the same `dataset/processed/train` and `dataset/processed/val` folders. It uses Adam, cross-entropy loss, and a ReduceLROnPlateau scheduler. The best checkpoint is selected by validation loss. The test set is not used for model selection. Default training length is 15 epochs and can be changed from the command line.

## GPU acceleration benchmark (Member 4 supplied measurements)
The supplied Member 4 report records a two-epoch benchmark with random initialization, batch size 16, 224 × 224 inputs, and NVIDIA T4 CUDA hardware.

| Model | Device | Training time (s) | Avg epoch (s) | Benchmark test accuracy | Inference latency (ms/image) | Throughput (images/s) | Peak GPU memory (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | CPU | 29.287 | 14.643 | 25.0% | 31.637 | 31.61 | N/A |
| CNN | CUDA | 3.573 | 1.786 | 25.0% | 0.475 | 2107.41 | 406.61 |
| ResNet18 | CPU | 48.577 | 24.288 | 42.5% | 47.246 | 21.17 | N/A |
| ResNet18 | CUDA | 1.763 | 0.881 | 25.0% | 0.799 | 1251.33 | 558.29 |

Reported training speedup:
- CNN: 8.197×
- ResNet18: 27.554×

These figures describe the supplied short hardware benchmark only. The benchmark models were initialized from scratch and trained for two epochs. Their accuracy values must not be reported as the final model accuracy. Runtime results are environment-dependent and should be reproduced on the target hardware before making general claims.

## Final results (15-epoch GPU run, 40 test images)
| Model | Training time (s) | Test accuracy | Macro-F1 |
|---|---:|---:|---:|
| Custom CNN | 12.55 | 100% | 1.00 |
| ResNet18 (random init) | 14.49 | 100% | 1.00 |

Final-checkpoint GPU inference (T4, forward pass, data preloaded): CNN up to 1529.7 images/s (batch 16, 0.654 ms/image); ResNet18 up to 1155.6 images/s (batch 32, 0.865 ms/image). Full table in `results/final_models_gpu_benchmark.csv`.

## Limitations
1. **Data leakage.** A near-duplicate audit (`results/near_duplicate_audit.json`) shows 35 of 40 test and 36 of 40 validation images have a train image with pixel correlation above 0.98. The exact-hash audit (0 duplicates) cannot detect this. The perfect scores therefore reflect memorisation, not clinical generalisation.
2. **Tiny test set** (40 images, one image = 2.5%).
3. **Data provenance.** The supplied images do not resemble real MRI slices and their source is undocumented; real-data performance is unknown.
4. The metrics file was transcribed from Colab output; `src/evaluate_final.py` regenerates it from checkpoints.
5. CPU and GPU inference timings were measured with different setups; no controlled speed-up is claimed. The Member 4 two-epoch benchmark (training speed-up CNN 8.2x, ResNet18 27.6x) is a separate hardware experiment, not a classification result.
6. Not a medical device; no clinical validation.

## Future work
Use a real, leakage-safe dataset (`src/prepare_real_dataset.py`), ImageNet-pretrained ResNet18 (`--pretrained`), patient-level splitting, cross-validation, and a controlled CPU-vs-GPU benchmark.

## Reproducibility and execution
Install dependencies:
```bash
pip install -r requirements.txt
```

Audit the provided processed dataset:
```bash
python src/audit_dataset.py
```

Train both models:
```bash
python src/train_all.py --epochs 15 --batch_size 16
```

Evaluate final models:
```bash
python src/evaluate_final.py
```

Audit expected handoff files:
```bash
python src/member5_audit.py
```

Expected data layout:
```text
dataset/processed/
  train/<class folders>/
  val/<class folders>/
  test/<class folders>/
```

Expected final checkpoints:
```text
models/cnn_model.pth
models/resnet18_model.pth
```

## Expected outputs
- `models/cnn_model.pth`
- `models/resnet18_model.pth`
- `results/cnn_training_history.json`
- `results/resnet18_training_history.json`
- `results/graphs/cnn_training_curves.png`
- `results/graphs/resnet18_training_curves.png`
- `results/final_evaluation.json`
- `results/classification_report.csv`
- `results/model_comparison.csv`
- `results/model_comparison.png`
- `results/confusion_matrices/cnn_confusion_matrix.png`
- `results/confusion_matrices/resnet18_confusion_matrix.png`

## Limitations
1. The complete original MRI dataset was not included in the received archive.
2. The included processed subset is small and is not sufficient by itself to support robust generalization claims.
3. The source README notes possible augmentation/oversampling in the combined dataset; dataset provenance should be verified.
4. Exact image-hash checks do not rule out transformed duplicates or patient-level leakage.
5. Model performance depends on split quality, class balance, image acquisition differences, random seed, and hardware.
6. This academic model is not clinically validated and must not be used for patient diagnosis or treatment decisions.

## Conclusion
The supplied project provides a reproducible starting point for four-class MRI image classification with a custom CNN and ResNet18, together with a separate CPU/GPU benchmark. The supplied benchmark indicates lower measured training time on the NVIDIA T4 GPU for both architectures. Final claims about classification quality remain pending until the full intended dataset and final trained model checkpoints are supplied and evaluated on a held-out test set.

## Team responsibilities
- Member 1: dataset collection, exploration, preprocessing, and split documentation.
- Member 2: custom CNN architecture and training.
- Member 3: ResNet18 implementation and training.
- Member 4: CPU/GPU benchmark and performance analysis.
- Member 5: data/model handoff audit, final held-out evaluation, metrics, confusion matrices, comparison, integration, and final reporting.
