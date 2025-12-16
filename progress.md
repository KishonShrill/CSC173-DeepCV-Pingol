# CSC172 Association Rule Mining Project Progress Report
**Student:** [Chriscent Louis June M. Pingol], [2022-0362]  
**Date:** [December 16, 2025]  
**Repository:** https://github.com/KishonShrill/CSC173-DeepCV-Pingol

## 📊 Current Status
| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | ✅ Completed | 2,527 images downloaded, split 80/20 |
| Architecture Selection | ✅ Completed | **Pivoted from ResNet50 to EfficientNetB0** |
| Initial Training | ✅ In Progress | Currently on Epoch 5 of 20 |
| Baseline Evaluation | ⏳ Pending | Awaiting final training run |
| Model Fine-tuning | ⏳ Not Started | Planned after baseline convergence |


## 1. Dataset Progress
- **Total images:** 2,527
- **Train/Val split: 80% Train (2,022 images) / 20% Validation (505 images)
- **Classes implemented: 6 classes: Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Preprocessing applied:
    - Resize: 384x384 (Increased from standard 224 for better detail)
    - Normalization: Scale 1./255
    - Augmentation: Rotation (30°), Zoom (0.2), Shear, Horizontal/Vertical Flip=


## 2. EDA Progress

**Architectural Pivot**: 
We initially proposed ResNet50. However, during early testing, we transitioned to EfficientNetB0.
- Reason: EfficientNetB0 offers a better parameter-to-accuracy ratio. It allows us to train with higher resolution images (384x384) to capture texture details (crucial for distinguishing paper vs. plastic) while maintaining a lighter memory footprint than ResNet.
- To have a comparative study if EfficientNetB0 performs better than ResNet50

**Current Metrics:**
| Metric | Value |
|--------|-------|
| None   | N/A   |

## 3. Challenges Encountered & Solutions
| Issue | Status | Resolution |
|-------|--------|------------|
| **Model Weight/Memory** | ✅ Fixed | **Switched from ResNet50 to EfficientNetB0** to allow for larger input resolution without exploding parameter count. |
| **CUDA Out of Memory** | ✅ Fixed | High resolution (384x384) caused OOM. **Reduced batch_size from 32 → 16.** |
| **Class Imbalance** | ⏳ Ongoing | The 'Trash' class has significantly fewer images (137). Applied **Class Weights** to the loss function during training. |
| **Overfitting** | ⏳ Monitoring | Applied heavy data augmentation (flips/zooms) and Dropout (0.2) in the top layers. |

## 4. Next Steps (Before Final Submission)
- [ ] Complete full 20-50 epoch training run with EfficientNetB0.
- [ ] Generate Confusion Matrix to analyze specific misclassifications (e.g., Glass vs. Plastic).
- [ ] Compare current results against the original ResNet50 baseline (if time permits).
- [ ] Record 5-min demo video showing the model predicting new images.
- [ ] Finalize README.md with comparison charts.
