# CNN Model (Member 2 Report Section)

## 1. Deep Learning and CNNs
Deep learning uses neural networks with many layers that learn features directly from data. A Convolutional Neural Network (CNN) is designed for images: instead of connecting every pixel to every neuron, it slides small learnable filters over the image, so the same filter detects a pattern (an edge, a texture, a region of brain shrinkage) anywhere in the scan.

Main building blocks:
- **Convolution layer:** applies learnable 3x3 filters and produces feature maps. Early layers capture edges and textures; deeper layers capture larger structures.
- **Batch normalisation:** normalises layer outputs, which stabilises and speeds up training.
- **Activation (ReLU):** f(x) = max(0, x). Adds non-linearity so the network can learn complex patterns.
- **Pooling (MaxPool 2x2):** halves the spatial size, reduces computation and gives some tolerance to small shifts.
- **Global average pooling:** turns each feature map into one number, which reduces parameters compared with flattening.
- **Dropout:** randomly switches off neurons during training to reduce overfitting.
- **Fully connected layers:** combine the learned features and output one score per class.

## 2. Architecture used
Input: 3 x 224 x 224 (grayscale MRI converted to 3 channels, resized and normalised by the data pipeline). Output: 4 classes (NonDemented, VeryMildDemented, MildDemented, ModerateDemented).

| Stage | Layers | Output shape | Parameters |
|---|---|---|---|
| Block 1 | Conv(3->32), BN, ReLU, MaxPool | 32 x 112 x 112 | 960 |
| Block 2 | Conv(32->64), BN, ReLU, MaxPool | 64 x 56 x 56 | 18,624 |
| Block 3 | Conv(64->128), BN, ReLU, MaxPool | 128 x 28 x 28 | 74,112 |
| Block 4 | Conv(128->256), BN, ReLU, MaxPool | 256 x 14 x 14 | 295,680 |
| Pooling | Global average pool | 256 | 0 |
| Classifier | Dropout(0.5), FC 256->128, ReLU, Dropout(0.3), FC 128->4 | 4 | 33,412 |
| **Total** | | | **422,788** |

Diagram: `results/graphs/cnn_architecture.png`.

## 3. Training methodology
- **Loss function:** Cross-entropy loss, the standard choice for multi-class classification.
- **Optimizer:** Adam, learning rate 0.001, with ReduceLROnPlateau (learning rate halves if validation loss does not improve for 2 epochs).
- **Batch size:** 32. **Epochs:** 15 (configurable).
- **Data:** the shared train/validation/test split prepared by Member 1 (70/15/15, stratified, duplicates removed, seed 42). Light augmentation (rotation up to 10 degrees) is applied to the training set only.
- **Model selection:** the weights with the best validation accuracy are saved to `models/cnn_model.pth`. The test set is used only once, at the end.
- **Monitoring:** training and validation loss and accuracy are recorded every epoch and plotted in `results/graphs/cnn_training_curves.png`.

## 4. Implementation
Framework: PyTorch. Files: `src/cnn_model.py` (model definition) and `src/train.py` (training loop, saving, plotting, timing). Results, including total training time and per-epoch time, are written to `results/cnn_results_<device>.json`.

## 5. Results (fill in after running)
| Item | Value |
|---|---|
| Device / hardware | |
| Total training time | |
| Best validation accuracy | |
| Test accuracy | |

All values must be copied from `results/cnn_results_*.json`, not estimated.

## 6. Likely viva questions
1. Why use BatchNorm and Dropout? (Stable training; reduce overfitting.)
2. Why global average pooling instead of flattening? (Far fewer parameters, less overfitting.)
3. What does the loss curve tell you if validation loss rises while training loss falls? (Overfitting.)
4. Why is the test set not used while training? (To give an unbiased estimate of performance.)
5. Why 3x3 filters and increasing channels (32 to 256)? (Small filters are efficient; deeper layers need more filters for richer features.)
