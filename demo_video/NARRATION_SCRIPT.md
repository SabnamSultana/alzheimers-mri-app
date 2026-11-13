# Narration script (walkthrough video, ~3.5 min)

## Slide 1
Group G eight. GPU accelerated deep learning based Alzheimer's disease classification from brain M R I images. This video walks through the project: the problem, the pipeline, the models, the results, the GPU benchmark and the prototype. This is an academic prototype and not a medical device.

## Slide 2
The goal is to classify a brain M R I slice into four stages: non demented, very mild, mild and moderate demented. We compare a small custom C N N with ResNet eighteen in PyTorch, measure how much a GPU speeds up training and inference compared with a C P U, and build a small upload and predict prototype.

## Slide 3
The pipeline loads the images, resizes them to two twenty four by two twenty four, trains the models on a GPU, evaluates on a held out test set, benchmarks the CPU against the GPU, and finally serves a demo application. We also run data integrity checks for duplicate images.

## Slide 4
These are real brain M R I slices from the M N I one five two template, which we use as demo inputs. They are not labelled patient scans. Images are resized to two twenty four by two twenty four before entering the network. A script is included to download a real labelled Alzheimer data set for the final training.

## Slide 5
The custom C N N has four convolution blocks with batch normalisation, global average pooling and a small classifier. ResNet eighteen is an eighteen layer residual network with a four output layer. Both are trained for fifteen epochs with the Adam optimiser on a Tesla T four GPU.

## Slide 6
Both models fit the training data well. ResNet eighteen converges much faster. The validation accuracy of the C N N fluctuates, because the validation set contains only forty images.

## Slide 7
On the forty image test set both models reach one hundred percent accuracy. However, our near duplicate audit shows that thirty five of the forty test images are near copies of training images. So this score reflects memorisation of a small easy data set, not real clinical performance.

## Slide 8
In the controlled benchmark, the GPU trained the C N N eight point two times faster and ResNet eighteen twenty seven point five times faster than the C P U. Inference was about sixty six times faster for the C N N and fifty nine times faster for ResNet eighteen. The GPU speeds up computation, but it does not by itself improve accuracy.

## Slide 9
The prototype is a Gradio web app. The user uploads an M R I image, chooses the C N N or ResNet eighteen, and gets the predicted stage, the class probabilities and the inference time. It runs on Colab or on a local machine with the commands shown. The confidence is a softmax score, not a medical probability.

## Slide 10
The main limitations are the tiny data set, only forty test images, near duplicate leakage, and that ResNet eighteen was not pretrained. The next step is to train on a real labelled data set with a leakage safe split, using the scripts we provide, then add pretrained weights, augmentation, cross validation and Grad CAM explanations.

## Slide 11
Thank you. This project is an academic demonstration only. It is not a medical device, it is not clinically validated, and it must not be used for diagnosis or treatment decisions.

Tip: for the final viva video, record your own screen (Colab: nvidia-smi, python src/demo_app.py with an upload) using DEMO_VIDEO_SCRIPT.md and these lines.
