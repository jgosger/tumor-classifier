import os
import cv2
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetConfig:
    data_dir: str
    img_size: int = 64
    grayscale: bool = True


def load_images_from_folder(folder_path, img_size, grayscale=True):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)

            if grayscale:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            images.append(img)

    return images


def load_dataset(config: DatasetConfig):
    no_path = os.path.join(config.data_dir, "no")
    yes_path = os.path.join(config.data_dir, "yes")

    no_images = load_images_from_folder(no_path, config.img_size, config.grayscale)
    yes_images = load_images_from_folder(yes_path, config.img_size, config.grayscale)

    X = no_images + yes_images
    y = ["no_tumor"] * len(no_images) + ["tumor"] * len(yes_images)

    X = np.array([img.flatten() for img in X], dtype=np.float32) / 255.0

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder