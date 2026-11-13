# ResNet18 Model (Member 3 Report Section)

## 1. Introduction
ResNet18 is a convolutional neural network architecture that uses residual connections. The shortcut connections help information and gradients travel through the network, making deeper models easier to optimize. In this project, ResNet18 is used as a second image-classification model for the Alzheimer's MRI dataset and its results can be compared with the CNN developed by Member 2.

## 2. Model architecture
The input image is resized to 224 × 224 pixels and converted to three channels by the shared preprocessing pipeline. ResNet18 extracts features through convolutional layers and residual blocks. Its final fully connected layer is replaced with a layer whose output count equals the number of dataset classes. The current project data description lists four classes: NonDemented, VeryMildDemented, MildDemented, and ModerateDemented. The actual folder names are read from the dataset at runtime and must be confirmed before final training.

## 3. Training methodology
- **Framework:** PyTorch and Torchvision.
- **Input size:** 224 × 224 pixels by default.
- **Loss function:** Cross-entropy loss for multi-class classification.
- **Optimizer:** Adam; default learning rate 0.0001.
- **Batch size:** 32 by default.
- **Epochs:** 15 by default; configurable from the command line.
- **Data split:** The script calls Member 1's `get_dataloaders()` function and reuses `results/split.csv`, so the CNN and ResNet18 should use the same train/validation/test split.
- **Model selection:** The checkpoint with the best validation accuracy is saved. The test set is used only after training for final evaluation.
- **Pretrained weights:** By default, the script starts with random weights so it can run without downloading weights. To use ImageNet transfer learning, run with `--pretrained`; this may require internet access the first time.
- **Hardware:** The script uses CUDA if selected and available; otherwise use CPU. Record the actual device used in the results.

## 4. Implementation files and outputs
- `src/resnet18_train.py`: training, validation, test evaluation, and result saving.
- `models/resnet18_model.pth`: best validation checkpoint.
- `results/graphs/resnet18_training_curves.png`: training/validation loss and accuracy curves.
- `results/graphs/resnet18_confusion_matrix.png`: confusion matrix for the held-out test set.
- `results/resnet18_results_<device>.json`: timing, accuracy, per-class precision/recall/F1, confusion matrix, and training history.

## 5. How to run
From the project root:

```bash
pip install -r requirements.txt
python src/resnet18_train.py --epochs 1
```

The one-epoch command is only a smoke test. If it completes successfully, run the intended experiment, for example:

```bash
python src/resnet18_train.py --epochs 15 --pretrained
```

If internet access is unavailable or pretrained weights cannot be downloaded, omit `--pretrained`. The dataset must be extracted to `dataset/combined_images/`, with one subfolder per class. Member 1's shared `results/split.csv` should be used. If that CSV contains paths from another computer or Colab session, regenerate it in the current project environment and share the regenerated CSV with the team.

## 6. Results (fill only after running on the real dataset)
| Metric | Measured value |
|---|---|
| Device / GPU name | To be measured |
| Number of classes | Read from dataset |
| Total training time | To be measured |
| Best validation accuracy | To be measured |
| Test accuracy | To be measured |
| Precision / recall / F1 per class | Generated in JSON output |

Do not enter estimated or invented performance numbers. The ZIP received from Members 1 and 2 did not include the actual MRI images or `results/split.csv`, so final metrics cannot be calculated until the dataset and split are available.

## 7. Important data limitation
The supplied dataset README describes a combined dataset of approximately 44,000 MRI slices and warns that it may be augmented or oversampled. The original dataset source and any augmentation/oversampling must be documented in the final report. If near-duplicate images or augmented versions of the same source image appear across training and test splits, test performance can be overly optimistic. Use a source-aware split if patient/source identifiers are available, and do not present this educational classifier as a clinical diagnostic tool.

## 8. Likely viva questions
1. **What is ResNet18?** A CNN architecture with 18 weighted layers and residual/shortcut connections.
2. **What is a residual connection?** A shortcut that adds a block's input to its transformed output, helping gradient flow.
3. **Why replace the final fully connected layer?** The original model's output count must match the classes in this dataset.
4. **What is transfer learning?** Reusing weights learned on a different dataset, then adapting the model to the target task.
5. **Why use validation data?** To monitor training and select a model without using the final test set.
6. **Why report precision, recall, and F1 as well as accuracy?** Class imbalance can make accuracy alone misleading.
7. **Why must CNN and ResNet18 use the same split?** It makes the comparison more fair and interpretable.
8. **Does high test accuracy prove clinical usefulness?** No. Clinical use requires appropriate patient-level validation, external testing, expert review, and regulatory/ethical safeguards.
