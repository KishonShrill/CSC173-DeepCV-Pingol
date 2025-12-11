# Garbage Classification Using ResNet50 (Deep Computer Vision Project)
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** [Chriscent Louis June M. Pingol], [2022-0362]
**Semester:** [AY 2025-2026 Sem 1]
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Installation](#installation)

## Introduction
### Problem Statement
Improper waste segregation is common in many local communities, including Iligan City. Manual sorting is slow, inconsistent, and exposes workers to health risks. This project proposes an automated deep learning image classifier for identifying waste types from images using a ResNet50 convolutional neural network.
The system predicts six classes: cardboard, glass, metal, paper, plastic, and trash, helping improve recycling processes and environmental sustainability.

### Objectives
- Develop an image classification model capable of classifying six garbage categories with competitive accuracy.
- Fine-tune a pre-trained ResNet50 architecture using transfer learning to achieve stable training and minimize overfitting.
- Implement a full training pipeline including dataset loading, preprocessing, augmentation, training, validation, testing, and visualization of accuracy/loss curves.
- Evaluate the model using standard metrics such as accuracy, confusion matrix, and F1-score.

![Problem Demo](images/problem_example.gif) [web:41]

## Methodology
### Dataset
- Source:
    - Name: Garbage Image Dataset — Kaggle (Farzad Nekouei)
    - Link: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
    - Estimated Size: 2,527 images, each ~512×384 resolution
- Split: 80/20/[real-world data] train/val/test
- Preprocessing: 
    - Augmentation 
    - Resizing to 384x384 [web:41]
    - Zoom
    - Flip
    - Rotation
    - Brightness Adjustment
    - Minor Width and Height Shift
    - Rescaling
    - Optional:
        - Channel Shift
        - Fill Mode

### Architecture
![Model Diagram](images/architecture.png)
- ResNet50 pretrained on ImageNet
- Frozen base layers
- Custom layers on top:
    - GlobalAveragePooling
    - Dense layers
    - Softmax output (6 units)

### Architecture Sketch
```scss
Input(224x224x3)
↓
ResNet50 (pretrained weights)
↓
GlobalAveragePooling
↓
Dropout(0.5)
↓
Dense(6) Softmax
```

## Installation
1. Clone repo: `git clone https://github.com/yourusername/CSC173-DeepCV-Pingol`
2. Install deps: `pip install -r requirements.txt`
3. Download weights: See `models/` or run `download_weights.sh` [web:22][web:25]

**requirements.txt:**
torch>=2.0
tensorflow
numpy
pandas
matplotlib
scikit-learn
seaborn
pillow
pydot
graphviz

4. Optional:
```bash
sudo apt install graphvis
# or
sudo dnf install graphvis
```

