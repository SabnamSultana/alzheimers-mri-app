# Run results and limitations

## Final run (used for report)
- Hardware: NVIDIA Tesla T4 (Google Colab), PyTorch CUDA build.
- Data: 160 train / 40 val / 40 test (4 classes, balanced), 224x224.
- CNN: 15 epochs, 12.55 s total. ResNet18 (random init): 15 epochs, 14.49 s total. Adam, ReduceLROnPlateau, best checkpoint by validation loss.
- Test (40 images): both models accuracy 1.00, macro-F1 1.00.

## Why these numbers must not be over-claimed
- 35/40 test images have a train image with pixel correlation > 0.98 (`results/near_duplicate_audit.json`).
- Train and validation accuracy also reach 100%; validation accuracy fluctuates (25%-100%) in early epochs, typical of a tiny, easy set.
- Images appear synthetic; real-data performance is unknown.

## GPU inference benchmark (final checkpoints, T4, forward pass only, data preloaded on GPU)
| Model | Batch | ms/image | images/s |
|---|---:|---:|---:|
| CNN | 1 | 0.7693 | 1299.96 |
| CNN | 16 | 0.6537 | 1529.68 |
| ResNet18 | 1 | 2.3544 | 424.74 |
| ResNet18 | 32 | 0.8654 | 1155.55 |
Full table: `results/final_models_gpu_benchmark.csv`. CPU (batch 1): CNN 20.27 ms, ResNet18 27.38 ms (`results/final_cpu_inference_benchmark.json`, measured earlier with a different setup, so not a controlled speed-up).

## Separate Member 4 hardware benchmark (2 epochs, random init, batch 16)
Training time CPU->CUDA: CNN 29.29 s -> 3.57 s (8.2x); ResNet18 48.58 s -> 1.76 s (27.6x). Their accuracies (25-42%) are benchmark artefacts, not final results.

## History
An earlier CPU-only run (CNN 3 epochs: 72.5%, ResNet18 1 epoch: 25%) was superseded by the GPU run above.
