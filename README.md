# Brain Tumor Classification (Classical ML Pipeline)

## Overview
This project builds and evaluates multiple machine learning models to classify brain MRI images as either:
- `tumor`
- `no_tumor` (healthy)

Models are compared using pipelines, hyperparameter tuning, and cross-validation to select the best-performing approach.

---

## Dataset

The dataset consists of labeled brain MRI images organized into two classes:

- **tumor**
- **no_tumor (healthy)**

Images are stored in the following structure:
```
data/
├── raw/
│   ├── tumor/
│   └── no_tumor/
```
The dataset is loaded, preprocessed, and split within the training pipeline.

## Why F1-Score?

Since medical classification problems often involve class imbalance and higher cost of false negatives (missing tumors), models were optimized using F1-score instead of accuracy.

---

## Key Skills Demonstrated

- Machine learning model comparison
- Hyperparameter tuning with GridSearchCV
- Cross-validation (5-fold)
- Data preprocessing for image classification
- Model persistence using Joblib