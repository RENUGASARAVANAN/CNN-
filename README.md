# Dog-Cat-Panda Classifier (PyTorch + CNN )

This project focuses on building an image classification model using a Convolutional Neural Network (CNN) implemented with PyTorch. The goal is to accurately classify animal images into one of three categories: dog, cat, or panda.

# Aim

To classify animal images into one of three categories: Dog, Cat, or Panda
To build and train a CNN model from scratch
To prevent overfitting using early stopping
To evaluate performance using accuracy and confusion matrix

# Dataset
We use the Dog-Cat-Panda dataset from Kaggle:

📎 Download Dataset from Kaggle
Format: image folders (train/, test/)
Classes: ['dog', 'cat', 'panda']
Train/Validation split is done manually (80/20)

# Model: CNN Architecture
```
Conv2d(3, 16, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)

Conv2d(16, 32, kernel_size=3, padding=1)
ReLU
MaxPool2d(2)

Flatten
Linear(32*32*32 → 128)
ReLU
Linear(128 → 3)  # 3 classes
```

Input image size: 128x128 Activation: ReLU Final output: 3 neurons (softmax-like output via CrossEntropyLoss) Optimizer: Adam (lr=0.0001) Loss Function: CrossEntropyLoss

# Steps Involved

Import libraries (torch, torchvision, sklearn, matplotlib)

Download dataset using kagglehub

Apply transforms: resize, normalize, augmentation (rotation, flip)

Split training set: 80% train, 20% validation

Define CNN model

Train with early stopping (based on validation accuracy)

Evaluate on test set

Visualize accuracy, loss, and confusion matrix

Show predictions with images

# Early Stopping

Monitors validation accuracy Stops training if no improvement for patience=5 epochs Restores best model weights automatically

# Evaluation Metrics

## Accuracy

The ratio of correctly predicted samples to the total samples: Accuracy = (Correct Predictions) / (Total Predictions).

# Confusion Matrix

<img width="647" height="695" alt="image" src="https://github.com/user-attachments/assets/54bf8d7f-6f0e-4a1d-8fe3-9edaac91f20d" />

# Sample Plots

<img width="1348" height="547" alt="image" src="https://github.com/user-attachments/assets/40026541-f398-4322-9b94-fbe2cba2077b" />

# How to Run
## 1.Install Dependencies
```
pip install -r requirements.txt
```
## 2.Run Training Script
```
python main.py
```

Ensure your dataset is downloaded to the correct path via kagglehub or placed in the appropriate train/ and test/ folders.

## Output:
<img width="492" height="747" alt="image" src="https://github.com/user-attachments/assets/c80dbc52-36ba-49f6-b025-819761b93018" />

Epoch [10/50], Loss: 0.82, Train Acc: 0.74, Val Acc: 0.78

Early stopping at epoch 15

Test Accuracy: 81.20%

# Result:
Thus,the project focuses on building an image classification model using a Convolutional Neural Network (CNN) implemented with PyTorch. The goal is to accurately classify animal images into one of three categories: dog, cat, or panda is executed successfully.

