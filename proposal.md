# CSC173 Deep Computer Vision Project Proposal
**Student:** [Chriscent Louis June M. Pingol], [2022-0362]  
**Date:** [December 11, 2025]

## 1. Project Title 
Deep Learning–Based Garbage Classification Using ResNet50

## 2. Problem Statement
Improper waste segregation is a persistent issue in many Philippine cities, including communities around Iligan and Mindanao State University–IIT. Manual sorting is time-consuming and prone to human error, which contributes to environmental degradation and inefficient recycling processes. This project aims to build an automated image-based garbage classification system using deep convolutional neural networks. By accurately identifying waste types from images, the system can support future smart recycling bins and local waste-management initiatives.

## 3. Objectives
- Develop an image classification model capable of classifying six garbage categories with competitive accuracy.
- Fine-tune a pre-trained ResNet50 architecture using transfer learning to achieve stable training and minimize overfitting.
- Implement a full training pipeline including dataset loading, preprocessing, augmentation, training, validation, testing, and visualization of accuracy/loss curves.
- Evaluate the model using standard metrics such as accuracy, confusion matrix, and F1-score.

## 4. Dataset Plan
- Source:
    - Name: Garbage Image Dataset — Kaggle (Farzad Nekouei)
    - Link: https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
    - Estimated Size: 2,527 images, each ~512×384 resolution
- Classes:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash
- Acquisition: Downloaded programmatically via:
```python
import kagglehub
path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")
```

## 5. Technical Approach
- Architecture sketch
    - Use ResNet50 pre-trained on ImageNet
    - Replace the final classification layer with a 6-class output layer
    - Train end-to-end with frozen early layers and fine-tuning later layers
- Model: ResNet50 (transfer learning, fine-tuned)
    - Chosen for performance and stability on small datasets
    - Strong feature extractor for cluttered object images like trash
- Framework: TensorFLow
- Hardware:
    - Google Collab GPU (T4 or L4)
    - Local GPU (GeForce GTX 1650)

## 6. Expected Challenges & Mitigations
- Challenge 1: Small Dataset → Risk of Overfitting
- Solution:
    - Data augmentation
    - Transfer learning with pre-trained weights
    - L2 regularization & dropout
    - Freeze early layers at start
- Challenge 2: Variability in Image Backgrounds
- Solution:
    - Strong augmentations
    - Fine-tuning deeper layers of ResNet to increase generalization
- Challenge 3: Class Imbalance (Trash only 137 samples)
- Solution:
    - Fixing Imbalance using class weights


