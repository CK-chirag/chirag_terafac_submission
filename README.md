# Flowers-102 Deep Learning Training Pipeline
**NOTE: I have runned the main ipynb file only, pls check for that run check**
A complete multi-level training framework built using **PyTorch**, designed for experimentation, performance improvement, and model interpretability on the **Flowers-102** classification dataset.

This project includes:
- Level 1: Baseline Transfer Learning (ResNet18)
- Level 2: Improved Training (Augmentation, Weight Decay, Scheduler)
- Level 3: Custom Architecture (ResNet34 + Custom Head)
- Level 4: Ensemble Learning (Averaging Predictions)
- Visual Testing (Sample Predictions Visualization)

---

## Project Structure

### **Level 1 — Baseline Model**
- Uses **ResNet18 pretrained on ImageNet**
- Replaces final layer → outputs **102 flower classes**
- Trains for 5 epochs
- Tracks:
  - Training loss  
  - Validation accuracy
- Saves model: `level1_beginner_model.pth`

### **Level 2 — Improved Training**
Enhancements:
- Data Augmentation:
  - Random Rotation
  - Horizontal Flip  
- Weight Decay (L2 Regularization)  
- Learning Rate Scheduler  
- Trains for 8 epochs  
- Saves model: `level2_model.pth`

### **Level 3 — Custom ResNet34 Model**
- ResNet34 backbone  
- Custom classification head:  
  - Linear → BatchNorm1d → ReLU → Dropout(0.5) → Linear  
- Trains for 10 epochs  
- Saves model: `level3_custom_model.pth`

### **Level 4 — Ensemble Learning**
- Loads Level 2 & Level 3 models  
- Makes predictions using:
