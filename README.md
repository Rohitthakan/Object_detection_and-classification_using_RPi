# Object Detection And Classification Using RPi

This project focuses on **classifying Laguerre–Gaussian (LG) laser modes** using a **custom 2D Convolutional Neural Network (CNN)**.  
The entire optical setup for data collection is **automated using Python and Raspberry Pi**, enabling the system to capture and classify LG modes efficiently — with potential for **real-time mode recognition**.

---

## 📘 Overview

The experiment involves generating LG modes by passing a **polarized laser through computer-generated holograms (CGHs)** corresponding to different LG modes.  
Each LG mode produces a unique intensity pattern, which is captured by a camera placed at the output of the optical setup.

Using this collected data, a custom CNN model is trained to classify the LG modes with near-perfect accuracy.

---

## ⚙️ Experimental Setup

- **Hardware Used:**
  - Laser source
  - Polarizer
  - Spatial Light Modulator (SLM) with holograms for LG modes
  - Raspberry Pi with Pi Camera
  - Optical elements and mirrors for alignment

- **Automation:**
  - The **entire optical setup communication** — including mode selection, camera capture, and data saving — is **automated using Python and Raspberry Pi**.
  - This allows large-scale image acquisition without manual intervention.

---

## 🧠 Dataset

| LG Mode | Total Images | Training | Validation | Testing |
|----------|---------------|-----------|-------------|----------|
| LG01     | 2000 | 1000 | 500 | 500 |
| LG02     | 2000 | 1000 | 500 | 500 |
| LG03     | 2000 | 1000 | 500 | 500 |
| LG04     | 2000 | 1000 | 500 | 500 |

- Images were captured automatically using the Raspberry Pi camera.
- Each image corresponds to a unique hologram pattern and laser configuration.

---

## 🧩 Model Architecture

A **custom 2D CNN** was built and trained using TensorFlow/Keras.  
The model includes:
- Multiple convolutional and pooling layers for feature extraction  
- Dropout layers for regularization  
- Fully connected dense layers for classification  

---

## 🚀 Training Details

- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 32  
- **Epochs:** 50  
- **Framework:** TensorFlow / Keras  
- **Training Data:** 1000 images per LG mode  
- **Validation Data:** 500 images per LG mode  

Training and validation progress is plotted and saved as:

- `training_accuracy.png`
- `training_loss.png`

---

## 📊 Results and Evaluation

### ✅ **Test Accuracy**
- Achieved **~100% classification accuracy** on test data.

### 🧾 **Confusion Matrix**
All test images were classified correctly except for **one image** of LG04, which was predicted as LG03.

| True \ Predicted | LG01 | LG02 | LG03 | LG04 |
|------------------:|:----:|:----:|:----:|:----:|
| **LG01** | 600 | 0 | 0 | 0 |
| **LG02** | 0 | 600 | 0 | 0 |
| **LG03** | 0 | 0 | 600 | 0 |
| **LG04** | 0 | 0 | 1 | 599 |

Confusion matrix is saved as `confusion_matrix.png`.

---

## 🎥 Real-Time Classification (Work in Progress)

- The trained CNN model **can be integrated** with a Raspberry Pi camera for **real-time LG mode classification**.
- A demo video of **real-time classification using YOLOv8 on Raspberry Pi** (custom dataset) is also included in the repository for reference.


