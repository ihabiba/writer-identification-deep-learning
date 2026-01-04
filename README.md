---

# Writer Identification using Deep Learning

This project implements a **deep learning–based writer identification system** using handwritten document images.
The goal is to predict the author of a handwritten page by learning discriminative handwriting patterns, **without using pretrained models and without resizing images**, following a strict geometry-preserving pipeline.

The system is designed to work with **very limited data** (one training page per writer) by decomposing each page into multiple informative patches and aggregating patch-level predictions at inference time.

---
## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-yellow?logo=scikit-learn)

---

## 🔍 Problem Overview

* **Task**: Offline handwritten writer identification
* **Input**: Scanned handwritten document images
* **Output**: Predicted writer ID
* **Constraints**:

  * No pretrained models
  * Deep learning only
  * Geometry-preserving preprocessing (no resizing)
  * Fully reproducible training and inference pipelines

---

## 🧠 Skills Demonstrated

- Deep Learning with Convolutional Neural Networks (CNNs)
- Geometry-preserving image preprocessing
- Patch-based learning for low-data regimes
- Model regularization (dropout, label smoothing, weight decay)
- Evaluation using Macro F1 and AUC
- Reproducible training and inference pipelines

---

## 📁 Project Structure

```text
writer-identification-deep-learning/
│
├── patching.py        # Geometry-preserving preprocessing and patch extraction
├── utils.py           # Shared utility functions
├── train.py           # Model training pipeline
├── run.py             # Inference and evaluation script
│
├── models/
│   └── writer_model.keras   # Trained CNN model
│
├── class_map.json     # Mapping between writer IDs and class indices
├── outputs/
│   └── result.csv     # Test predictions
│
├── requirements.txt
└── README.md
```

---

## 🧩 Preprocessing & Patch Extraction (`patching.py`)

To overcome the lack of training data at the page level, each handwritten page is decomposed into multiple patches.

**Key steps:**

* Convert images to grayscale
* Apply **Otsu binarization** to separate ink from background
* Detect text lines using **horizontal projection + morphological dilation**
* Perform **tight horizontal cropping**
* Enforce fixed patch size **only using cropping and padding (no resizing)**
* Filter patches using an **ink ratio threshold**
* Apply a fallback strategy if no text lines are detected

**Patch size**: `128 × 256`
This preserves handwriting geometry and avoids distortion-based accuracy gains.

---

## 🧠 Feature Learning Strategy

* No handcrafted features are used
* Feature learning is performed entirely by the CNN
* The only form of feature engineering is **patch-level decomposition**
* Light photometric augmentation (brightness & contrast) is applied during training

---

## 🏗️ Network Architecture (`train.py`)

A **deep CNN trained from scratch** is used:

* **5 convolutional blocks**
  Filters: `32 → 64 → 128 → 256 → 512`
  Each block includes:

  * 3×3 Convolution
  * Batch Normalization
  * ReLU activation
  * Max Pooling

* **Global Average Pooling**

* Fully connected head:

  * Dense(512) → BatchNorm → Dropout(0.5)
  * Dense(256) → Dropout(0.4)

* Output layer: Softmax over 70 writers

**Training details:**

* Optimizer: AdamW (with weight decay)
* Loss: Categorical Cross-Entropy with label smoothing
* Early stopping and learning-rate scheduling applied
* Patch count capped per writer to prevent overfitting

---

## 🚀 Inference & Evaluation (`run.py`)

* Uses **exactly the same preprocessing** as training
* Each page is decomposed into patches
* Patch-level probabilities are **aggregated (summed)** to obtain a page-level prediction
* This improves robustness and reduces sensitivity to noisy patches

### Evaluation Metrics

* **Top-1 Accuracy**
* **Top-3 / Top-5 Accuracy**
* **Macro F1 Score**
* **Macro AUC (One-vs-Rest)**

Top-K metrics are included to show ranking quality, which is valuable in forensic or investigative scenarios.

---

## 📊 Final Results

| Metric         | Value      |
| -------------- | ---------- |
| Top-1 Accuracy | **87.14%** |
| Top-3 Accuracy | 93.57%     |
| Top-5 Accuracy | 95.71%     |
| Macro F1 Score | 0.8439     |
| Macro AUC      | 0.9943     |

These results demonstrate strong generalization despite having **only one training page per writer**.

---

## 📦 Reproducibility

To run inference:

```bash
python run.py
```

Training can be reproduced with:

```bash
python train.py
```

Required dependencies are listed in `requirements.txt`.

---

## ✅ Key Takeaways

* Patch-based learning effectively overcomes extreme data scarcity
* Geometry-preserving preprocessing is critical for handwriting analysis
* A deeper CNN with proper regularization significantly improves performance
* Patch aggregation yields stable and reliable writer predictions

---
