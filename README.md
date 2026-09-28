# 🗑️ Garbage Classification using ResNet from Scratch

A deep learning project for **garbage image classification** using a custom **ResNet-inspired Convolutional Neural Network built from scratch** with PyTorch.

The model is trained to classify waste images into multiple categories and is deployed with a simple **Gradio web interface** that allows users to upload an image and receive the predicted class with confidence scores.

---

## 📌 Project Overview

Waste classification is an important application of computer vision that can help improve waste management and recycling systems.

In this project, a custom Residual Network is implemented **from scratch without using transfer learning**. The model learns visual features directly from the garbage image dataset.

The complete workflow includes:

* Dataset extraction and preparation
* Image preprocessing
* Data augmentation
* Train/Validation/Test splitting
* Custom ResNet architecture
* Model training using PyTorch
* Model evaluation
* Classification report
* Confusion matrix
* Saving the trained model
* Interactive image classification using Gradio

---

## 🧠 Model Architecture

The project uses a custom **ResNet-style architecture** implemented using PyTorch.

### Main Components

```text
Input Image
     │
     ▼
Convolution + Batch Normalization + ReLU
     │
     ▼
Max Pooling
     │
     ▼
Residual Block 1
     │
     ▼
Residual Block 2
     │
     ▼
Residual Block 3
     │
     ▼
Residual Block 4
     │
     ▼
Adaptive Average Pooling
     │
     ▼
Dropout
     │
     ▼
Fully Connected Layer
     │
     ▼
Garbage Class
```

The network contains residual connections that allow information to bypass convolutional layers, following the main idea behind ResNet architectures.

### Residual Block

Each residual block contains:

* 3×3 Convolution
* Batch Normalization
* ReLU activation
* 3×3 Convolution
* Batch Normalization
* Shortcut connection
* ReLU activation

---

## 📂 Dataset

The dataset is loaded using PyTorch's `ImageFolder`:

```python
torchvision.datasets.ImageFolder
```

The notebook automatically detects the directory containing the image classes.

The dataset is divided into:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

A fixed random seed (`42`) is used to make the dataset split reproducible.

---

## 🖼️ Image Preprocessing

All images are resized to:

```text
128 × 128 pixels
```

### Training Augmentation

The training images use:

* Random Horizontal Flip
* Random Rotation up to 15°
* Random Brightness adjustment
* Random Contrast adjustment
* Conversion to Tensor

Example:

```python
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
])
```

Validation and test images are resized and converted to tensors without random augmentation.

---

## ⚙️ Training Configuration

The model is trained using:

| Parameter     |              Value |
| ------------- | -----------------: |
| Image Size    |          128 × 128 |
| Batch Size    |                 32 |
| Epochs        |                 30 |
| Optimizer     |               Adam |
| Learning Rate |              0.001 |
| Weight Decay  |             0.0001 |
| Loss Function | Cross Entropy Loss |
| Dropout       |                0.4 |
| Random Seed   |                 42 |

The model automatically uses GPU when CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 📊 Model Evaluation

After training, the model is evaluated using the test dataset.

The project generates:

### Accuracy

```python
accuracy_score(y_true, y_pred)
```

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)
```

### Confusion Matrix

A confusion matrix is generated to visualize the model's classification performance across the different garbage categories.

---

## 💾 Saved Model

After training, the model weights are saved as:

```text
garbage_resnet_scratch.pt
```

The class names are also saved using Pickle:

```text
class_names.pkl
```

These files can be used later to load the trained model and perform predictions without retraining it.

---

## 🌐 Gradio Interface

The project includes a simple interactive interface using **Gradio**.

Users can:

1. Upload a garbage image.
2. The image is resized to 128×128.
3. The trained model processes the image.
4. The model predicts the garbage category.
5. Prediction probabilities are displayed.

The interface is created using:

```python
gr.Interface()
```

Example:

```text
┌──────────────────────────────────────┐
│        🗑️ Garbage Classification     │
├──────────────────────────────────────┤
│                                      │
│       Upload a garbage image         │
│                                      │
│              [ Image ]               │
│                                      │
├──────────────────────────────────────┤
│ Prediction:                          │
│                                      │
│ Plastic       ████████████  85%      │
│ Paper         ██            10%      │
│ Glass         █              5%      │
│                                      │
└──────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **PIL**
* **Gradio**
* **Google Colab**

---

## 📁 Project Structure

```text
GarBage_Classification/
│
├── GarBage_Classification.ipynb
├── garbage_resnet_scratch.pt
├── class_names.pkl
└── README.md
```

> The dataset is not included in this repository because the notebook loads it from Google Drive.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/rahafalnjjar49-cloud/GarBage_Classification.git
```

### 2. Open the notebook

You can run the project using:

* Google Colab
* Jupyter Notebook

The notebook also contains an **Open in Colab** button.

### 3. Prepare the dataset

The notebook expects the dataset ZIP file to be available in Google Drive:

```text
MyDrive/datasets/archive(8).zip
```

### 4. Run the notebook

Run the cells in order:

```text
Dataset Preparation
        ↓
Preprocessing
        ↓
Dataset Splitting
        ↓
Model Creation
        ↓
Training
        ↓
Evaluation
        ↓
Model Saving
        ↓
Gradio Interface
```

### 5. Upload an image

After training, the Gradio interface will open and allow you to upload a garbage image for classification.

---

## 🔬 Key Features

* ✅ Custom ResNet architecture
* ✅ Built from scratch
* ✅ No transfer learning
* ✅ Data augmentation
* ✅ 70/15/15 dataset split
* ✅ GPU support
* ✅ Classification report
* ✅ Confusion matrix
* ✅ Model saving
* ✅ Interactive Gradio interface
* ✅ Prediction confidence scores

---

## 🎯 Project Goal

The main goal of this project is to demonstrate how a **deep convolutional neural network with residual connections** can be developed from scratch and applied to a real-world computer vision problem: **automatic garbage classification**.

The project also demonstrates the complete machine learning pipeline from dataset preparation and model training to evaluation and deployment.

---

## 👩‍💻 Author

**Rahaf Medhat Al-Najjar**

AI Engineer



## 📜 License

This project is intended for educational and research purposes.
