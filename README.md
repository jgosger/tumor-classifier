# Brain Tumor Classification (Classical ML Pipeline)

## Overview
This project builds and evaluates multiple machine learning models to classify brain MRI images as either:
- `tumor`
- `no_tumor` (healthy)

Models are compared using pipelines, hyperparameter tuning, and cross-validation to select the best-performing approach.

---

## Dataset
Folder structure:

tumor-classifier/
├── data/
│   └── raw/
│       └── Tumor/
│           ├── yes/
│           └── no/
├── models/
│   ├── best_model.pkl
│   └── label_encoder.pkl
├── reports/
│   └── metrics.json
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
└── README.md


## Why F1-Score?

Since medical classification problems often involve class imbalance and higher cost of false negatives (missing tumors), models were optimized using F1-score instead of accuracy.

---

## Key Skills Demonstrated

- Machine learning model comparison
- Hyperparameter tuning with GridSearchCV
- Cross-validation (5-fold)
- Data preprocessing for image classification
- Model persistence using Joblib