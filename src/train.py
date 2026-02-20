import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from data import DatasetConfig, load_dataset
from model import get_model_and_param_grid


def main():

    # -------------------------
    # Load Data
    # -------------------------
    data_path = os.path.join("data", "raw", "Tumor")
    config = DatasetConfig(data_dir=data_path, img_size=64)

    X, y, label_encoder = load_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models_to_try = ["knn", "logreg", "dtree"]

    best_model = None
    best_score = -1
    best_model_name = None
    results_summary = {}

    # -------------------------
    # Model Comparison Loop
    # -------------------------
    for model_name in models_to_try:

        print(f"\n=== Training {model_name.upper()} ===")

        pipeline, param_grid = get_model_and_param_grid(model_name)

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="f1",  # optimize tumor detection
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Best Params:", grid.best_params_)
        print("Accuracy:", round(accuracy, 4))
        print("F1 Score:", round(f1, 4))
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        results_summary[model_name] = {
            "best_params": grid.best_params_,
            "accuracy": float(accuracy),
            "f1_score": float(f1)
        }

        if f1 > best_score:
            best_score = f1
            best_model = best_estimator
            best_model_name = model_name

    # -------------------------
    # Save Best Model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # Save metrics
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)

    print("\n====================================")
    print("Best Model:", best_model_name)
    print("Best F1 Score:", round(best_score, 4))
    print("Model saved to models/best_model.pkl")
    print("Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()