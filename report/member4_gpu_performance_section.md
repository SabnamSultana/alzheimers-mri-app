# Member 4 – GPU Acceleration and Performance Analysis

## Objective
Compare the execution performance of a custom CNN and ResNet18 on CPU and NVIDIA CUDA GPU. This section reports the benchmark run stored in `results/member4_benchmark_results.json` and `results/member4_performance_comparison.csv`.

## Experimental setup
- Dataset: the project’s supplied pre-split MRI image folders (`dataset/processed/train`, `val`, and `test`).
- Classes: four Alzheimer’s MRI image categories.
- Input size: 224 × 224 RGB tensors (grayscale images converted to three channels).
- Batch size: 16.
- Benchmark training: 2 epochs per model/device, from random initialization.
- Framework: PyTorch / Torchvision.
- GPU: NVIDIA T4 (CUDA).

## Measured results

| Model | Device | Training time (s) | Avg. epoch (s) | Test accuracy (%) | Inference latency (ms/image) | Throughput (images/s) | Peak GPU memory (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | CPU | 29.287 | 14.643 | 25.0 | 31.637 | 31.61 | N/A |
| CNN | CUDA | 3.573 | 1.786 | 25.0 | 0.475 | 2107.41 | 406.61 |
| ResNet18 | CPU | 48.577 | 24.288 | 42.5 | 47.246 | 21.17 | N/A |
| ResNet18 | CUDA | 1.763 | 0.881 | 25.0 | 0.799 | 1251.33 | 558.29 |

## Training speedup
- **CNN: 8.197×** (CPU training time / CUDA training time).
- **ResNet18: 27.554×** (CPU training time / CUDA training time).

## Interpretation and limitations
The recorded run shows lower training wall-clock time on CUDA than CPU for both benchmark models. These measurements describe a short, two-epoch hardware benchmark only. Models were initialized from scratch; therefore, the reported validation/test accuracies are **not** the final project model performance and must not be presented as the final Alzheimer’s classification accuracy. The short run is also too small to establish robust generalization. CPU and GPU timing can vary with Colab runtime load, data-loader overhead, and hardware availability.

The benchmark inference latency is measured over batches and is hardware-dependent. Peak CUDA memory is only available for the GPU run; CPU memory was not measured.

## What Member 5 should do next
1. Use the trained project models from Members 2 and 3, not the random-initialization benchmark weights, for final evaluation.
2. Evaluate both models on the same held-out `dataset/processed/test` split.
3. Generate confusion matrices, class-wise precision/recall/F1, and overall accuracy.
4. Keep benchmark performance metrics separate from final classification metrics.
5. Add final evaluation results to the main report after reproducing and verifying them.
