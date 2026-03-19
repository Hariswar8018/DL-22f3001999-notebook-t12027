# 🎧 Audio Classification — Deep Learning & GenAI Project
### IIT Madras | Roll No: `22f3009991` | Name: **Anshuman Samasi**

---

> 📄 **[Project Report](https://drive.google.com/demo.pdf)** &nbsp;|&nbsp; 🤗 **[HuggingFace Demo](https://hf.co/demo)**

---

## 📌 Project Overview

This project was developed as part of the **IIT Madras Deep Learning & Generative AI** course. The goal was to build and compare multiple deep learning architectures for audio classification, leveraging data augmentation and pre-trained transformer models to maximize performance.

---

## 🏆 Model Performance Summary

| Model | F1 Score | Type |
|-------|----------|------|
| CNN | 0.50 | Baseline |
| BiLSTM | 0.70 | Sequential |
| **AST (Pre-trained)** | **0.80** | **Best Model ✅** |

> The **Audio Spectrogram Transformer (AST)** pre-trained model achieved the highest F1 score of **0.80**, outperforming both the CNN baseline and BiLSTM model significantly.

---

## 🗂️ Models Used

### 1. 🔷 CNN (Convolutional Neural Network)
- Served as the **baseline model**
- Applied convolutional layers over mel-spectrogram features
- **F1 Score: 0.50**

### 2. 🔶 BiLSTM (Bidirectional Long Short-Term Memory)
- Captured **temporal dependencies** in both forward and backward directions
- Better suited for sequential audio features
- **F1 Score: 0.70**

### 3. ⭐ AST — Audio Spectrogram Transformer (Pre-trained)
- Leveraged a **pre-trained transformer** fine-tuned on audio spectrograms
- Transfer learning from large-scale audio datasets
- Best generalization and classification performance
- **F1 Score: 0.80**

---

## 🔧 Data Augmentation Techniques

A rich set of data augmentation strategies was applied to improve model robustness and generalization:

- 🔊 **Time Stretching** — Alter audio playback speed without changing pitch
- 🎵 **Pitch Shifting** — Shift audio pitch up or down
- 🌫️ **Gaussian Noise Addition** — Inject random noise for robustness
- ✂️ **Time Masking** — Randomly mask time steps in spectrograms (SpecAugment)
- 📏 **Frequency Masking** — Mask frequency bands in spectrograms (SpecAugment)
- 🔁 **Random Cropping** — Extract random audio segments
- 🔈 **Volume Scaling** — Randomly scale amplitude levels
- 🌀 **Mixup Augmentation** — Blend two audio samples and their labels
- 🎛️ **Background Noise Overlay** — Mix in real-world background sounds

---

## 📈 Milestone Tracker

| Milestone | Commit | Description |
|-----------|--------|-------------|
| Milestone 1 | `Commit 17` | Initial setup, data pipeline & baseline CNN |
| Milestone 2 | `Commit 19` | BiLSTM implementation & evaluation |
| Milestone 3 | `Commit 20` | AST pre-trained model integration |
| Milestone 4 | `Commit 23` | Data augmentation pipeline & fine-tuning |
| Milestone 5 | `Commit 26` | Final evaluation, deployment & report |

---

## 🚀 Deployment

The best-performing **AST model** has been deployed on HuggingFace Spaces for live inference.

🔗 **Live Demo:** [hf.co/demo](https://hf.co/demo)

---

## 📎 Links

| Resource | Link |
|----------|------|
| 🤗 HuggingFace Demo | [hf.co/demo](https://hf.co/demo) |
| 📄 Project Report | [drive.google.com/demo.pdf](https://drive.google.com/demo.pdf) |

---

## 👤 Author

| Field | Details |
|-------|---------|
| **Name** | Anshuman Samasi |
| **Roll No** | 22f3009991 |
| **Institution** | IIT Madras |
| **Course** | Deep Learning & Generative AI |

---

*Submitted as part of the IIT Madras Online Degree Programme — Deep Learning & GenAI Track.*
