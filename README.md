## Overview

-  Multi-label classification (103 categories, ~804K documents)
-  GPU-accelerated 3-layer DNN with Batch Normalization
-  5 baseline model comparisons (Logistic, SVM, Random Forest, etc.)
-  8+ evaluation metrics with rich visualizations
-  **5.5%+ F1-score improvement** over traditional methods

## Performance

| Model | F1 (Micro) | Precision | Recall | Subset Accuracy |
|-------|------------|-----------|--------|-----------------|
| **DNN (Ours)** | **0.8415** | **0.8679** | **0.8166** | **0.6000** |
| Logistic Regression | 0.7975 | 0.9278 | 0.6993 | 0.5073 |
| Decision Tree | 0.7340 | 0.7937 | 0.6826 | 0.3646 |
| SGD Classifier | 0.7081 | 0.9515 | 0.5639 | 0.3690 |


## Model Architecture

```
Input (47,236) → Dense(512) → BatchNorm → ReLU → Dropout(0.3)
               → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
               → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
               → Dense(103) → Sigmoid
```

**Config**: Adam optimizer (lr=0.001), batch size=512, 20 epochs, BCEWithLogitsLoss

## Key Features

- **GPU Support**: Automatic CUDA detection
- **Sparse Matrix**: Memory-efficient data handling
- **Rich Metrics**: F1, Precision, Recall, Jaccard, Hamming Loss, Subset Accuracy
- **Visualizations**: 12+ plots (training history, confusion matrices, per-label analysis)
- **Model Persistence**: Save/load trained models

##  Project Structure

```
├── main.py                   # Main training script
├── model.py                  # DNN architecture
├── data_loader.py            # Data preprocessing
├── evaluator.py              # Evaluation metrics
├── baseline_models.py        # Traditional ML models
├── compare_models.py         # Model comparison
├── test_model.py             # Batch testing
├── data/                     # Dataset storage
├── models/                   # Saved models
├── results/plots/            # Visualizations
└── comparison_results/       # Comparison reports
```

# Multi-Label Classification Project Report

This document provides a comprehensive report on the configuration of all models used in the multi-label classification project.

## 1. Dataset
*   **Dataset:** RCV1 (Reuters Corpus Volume I)
*   **Type:** Multi-label classification
*   **Features:** TF-IDF vectors (47,236 dimensions)
*   **Categories:** 103 topics
*   **Split:** Train (70%), Validation (10%), Test (20%)

## 2. Model Configurations

The following models were trained and evaluated using the `compare_models.py` script.

### 2.1. Logistic Regression (Baseline)
*   **Library:** Scikit-learn
*   **Wrapper:** `OneVsRestClassifier`
*   **Solver:** `lbfgs`
*   **Max Iterations:** 1000
*   **Random State:** 42
*   **Parallelism:** `n_jobs=-1` (All available cores)

### 2.2. SGD Classifier (Baseline)
*   **Library:** Scikit-learn
*   **Wrapper:** `OneVsRestClassifier`
*   **Loss Function:** `log_loss` (Logistic Regression via SGD)
*   **Max Iterations:** 1000
*   **Random State:** 42
*   **Parallelism:** `n_jobs=-1`

### 2.3. Random Forest (Baseline)
*   **Library:** Scikit-learn
*   **Wrapper:** `MultiOutputClassifier`
*   **Number of Estimators:** 100
*   **Max Depth:** 20
*   **Random State:** 42
*   **Parallelism:** `n_jobs=-1`

### 2.4. Naive Bayes (Baseline)
*   **Library:** Scikit-learn
*   **Wrapper:** `MultiOutputClassifier`
*   **Type:** `MultinomialNB`
*   **Alpha (Smoothing):** 1.0
*   **Parallelism:** `n_jobs=-1`

### 2.5. Decision Tree (Baseline)
*   **Library:** Scikit-learn
*   **Wrapper:** `MultiOutputClassifier`
*   **Max Depth:** 20
*   **Random State:** 42
*   **Parallelism:** `n_jobs=-1`

### 2.6. Deep Neural Network (Our Model)
*   **Framework:** PyTorch
*   **Architecture:**
    *   **Input Layer:** 47,236 features (RCV1 TF-IDF)
    *   **Hidden Layer 1:** 512 neurons, Batch Normalization, Leaky ReLU, Dropout (0.5)
    *   **Hidden Layer 2:** 128 neurons, Batch Normalization, Leaky ReLU, Dropout (0.5)
    *   **Output Layer:** 103 neurons (one per class)
*   **Activation Function:** Leaky ReLU
*   **Dropout Rate:** 0.5
*   **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy with Logits)
    *   **Class Balancing:** Weighted loss using `pos_weight = sqrt(n_negatives / n_positives)` to handle class imbalance.
*   **Optimizer:** Adam
    *   **Learning Rate:** 0.001
    *   **Weight Decay:** 1e-4 (L2 Regularization)
*   **Training Configuration:**
    *   **Batch Size:** 512
    *   **Epochs:** 50
    *   **Device:** GPU (if available)
*   **Prediction Threshold:** 0.5

## 3. Evaluation Metrics
All models are evaluated on the Test set using the following metrics:
*   **F1 Score:** Micro, Macro, and Weighted averages.
*   **Precision:** Micro and Macro averages.
*   **Recall:** Micro and Macro averages.
*   **Subset Accuracy:** Exact match ratio.
*   **Hamming Loss:** Fraction of incorrect labels.
*   **Jaccard Score:** Intersection over Union of predicted vs true labels.

## 4. File Structure
*   `main.py`: Trains the DNN model.
*   `compare_models.py`: Trains baseline models and compares them with the pre-trained DNN.
*   `model.py`: Defines the DNN architecture and the `BaselineModel` wrapper.
*   `data_loader.py`: Handles downloading and splitting the RCV1 dataset.
*   `evaluator.py`: Contains metrics calculation and plotting functions.
*   `baseline_models.py`: Implementation of baseline models using scikit-learn.


