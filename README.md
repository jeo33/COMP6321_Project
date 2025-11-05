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



