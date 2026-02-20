# Brain Tumor Classification (Classical ML Pipeline)

## Overview
This project compares multiple machine learning models to classify brain MRI images as either:
- `tumor`
- `no_tumor` (healthy)

Models were compared using pipelines and hyperparameter tuning to select the best-performing approach.

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
## Results

Models were compared using 5-fold cross-validation and optimized for F1-score.

- **Best Model:** Logistic Regression (tuned with GridSearchCV)
- **Best F1 Score:** 0.836
- **Test Accuracy:** 0.784

The final model was saved using joblib for reproducibility.
## Why F1-Score?

Since medical classification problems often involve class imbalance and higher cost of false negatives (missing tumors), models were optimized using F1-score instead of accuracy.

## Key Skills Demonstrated

- Machine learning model comparison
- Hyperparameter tuning with GridSearchCV
- Cross-validation (5-fold)
- Data preprocessing for image classification
- Model persistence using Joblib