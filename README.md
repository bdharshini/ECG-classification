# ECG Classification

Evaluating different models on classifying ECG data. The best model is further optimized using metaheuristic optimizers.

---

## 📄 Reference Paper

This project is an implementation of the paper:  
*"Optimizing machine learning for enhanced automated ECG analysis in cardiovascular healthcare"*  
[ScienceDirect Link](https://www.sciencedirect.com/science/article/pii/S1110866524001415)

---

## 📊 Dataset

Dataset was obtained from PhysioNet's MIT-BIH Arrhythmia Database:  
[MIT-BIH Dataset Link](https://physionet.org/content/mitdb/1.0.0/)  

**Note:** The dataset is not included in this repository. You can download it directly from the link above.

---

## 📝 Description

Current care in cardiovascular ailments is transforming in the era of connected health technology, with devices like wearables generating vast amounts of ECG data that require accurate and deep interpretation.

Many traditional ECG analysis algorithms rely on heuristic feature extraction with shallow architectures and suffer from poor classification performance. This study addresses the challenge of developing a robust and efficient automated ECG classification system using machine learning (ML) and optimization techniques.

---

## ⚖️ Class Imbalance Handling

The dataset is highly imbalanced. The following techniques were used and compared to find the best fit:  
1. GAN (Generative Adversarial Network)  
2. SMOTE (Synthetic Minority Over-sampling Technique)  
3. Class weights adjustment

---

## 🧩 Models Used

Initial model parameter tuning was performed using Optuna for the following classifiers:  
- Random Forest  
- XGBoost  
- SVC (Support Vector Classifier)  
- LinearSVC

---

## 🔍 Metaheuristic Optimizers Used
- JADE  
- EnhancedAEO  
- OriginalJA  
- LevyJA
Note:All metaheuristic optimizers are present in the `mealpy` library. However, custom functions for LevyJA and OrginalJA were written instead of using the lib.
ref: [Paper Link](https://www.sciencedirect.com/science/article/abs/pii/S0957417420306989)
---

## 📈 Results

The combination **XGBoost + class weights + JADE optimizer** achieved the best performance with a final macro F1-score of **0.9526**.

---

## ⚠️ Limitations

1. Restricted computational resources limited optimizer population size and number of iterations.  
2. Results validated on a single dataset; broader validation is needed.  
3. The dataset has a significant class imbalance which remains challenging.

---

