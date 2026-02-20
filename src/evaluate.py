import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data import DatasetConfig, load_dataset


def main():
    # Load model + encoder
    model = joblib.load("models/knn_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    # Load data
    data_path = os.path.join("data", "raw", "Tumor")
    config = DatasetConfig(data_dir=data_path, img_size=64)

    X, y, _ = load_dataset(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()