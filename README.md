# Pneumonia Detection from Chest X-ray Images using PyTorch CNN Model Architecture
**Disclaimer**: This notebook is adapted from <a href="https://medium.com/artificial-corner/chest-x-ray-pneumonia-detection-65315fd4e186">“Chest X-ray Pneumonia Detection” by Golnaz Hosseini (Artificial Corner)</a>.

The notebook uses a pre-trained CNN model ('DenseNet121') to classify Chest X-ray images into normal and pneumonia.

## Task Details
### Objective
Build a CNN model to classify chest X-ray images into pneumonia and normal categories.

### Dataset
<a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Chest X-Ray Images (Pneumonia) dataset from Kaggle</a>

## Structure of the Notebook
### Data Preprocessing
Resizing, normalization, and data augmentation

### CNN Model Architecture
Using PyTorch library and a pre-trained CNN model ('DenseNet121')

### Evaluation
Assess the model using 4 metrics - accuracy, precision, recall, F1-score, and also Confusion matrix

### Visualizations
(i) Training/validation loss and accuracy plots \
(ii) Receiver Operating Characteristic (ROC) curve

## Quick Questions
### Why Pre-Trained Model?
The DenseNet121 model has been trained on the ImageNet dataset, which contains over 1 million images. It can accurately recognise a wide variety of objects, helping save time and effort in training.

### Why PyTorch?
The PyTorch library is used to implement various functions in the notebook. One of which is 'Torchvision' --- part of the Pytorch library for Computer Vision with a wide range of image transformation functions.
