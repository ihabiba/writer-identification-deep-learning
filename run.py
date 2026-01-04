import os
import json
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from patching import PatchConfig, extract_line_patches_from_gray
from utils import is_png, load_gray_image


# Paths
TEST_DIR = "dataset/test"
MODEL_PATH = "models/writer_model.keras"
CLASS_MAP_PATH = "class_map.json"
OUT_CSV = "outputs/result.csv"


# Patch configuration (must match training)
CFG = PatchConfig(
    patch_h=128,
    patch_w=256,
    seg_per_line=8,
    min_ink_ratio=0.01,
)


# Class map loading
def load_class_map(path: str):
    with open(path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


# Patch normalization before inference
def preprocess_patch_numpy(patch: np.ndarray) -> tf.Tensor:
    x = patch.astype(np.float32) / 255.0
    x = 1.0 - x
    x = x[..., None]
    return tf.convert_to_tensor(x)


# Page-level prediction by aggregating patch probabilities
def predict_page_probs(
    model: keras.Model,
    gray: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    patches = extract_line_patches_from_gray(gray, CFG)

    if len(patches) == 0:
        raise RuntimeError("No patches extracted from test image.")

    probs_sum = np.zeros((num_classes,), dtype=np.float32)

    for p in patches:
        x = preprocess_patch_numpy(p)
        x = tf.expand_dims(x, 0)
        probs = model.predict(x, verbose=0)[0]
        probs_sum += probs

    return probs_sum


# Evaluation loop and metric computation
def main():
    model = keras.models.load_model(MODEL_PATH)
    class_to_idx, idx_to_class = load_class_map(CLASS_MAP_PATH)
    num_classes = len(class_to_idx)

    y_true, y_pred = [], []
    y_true_idx, y_score = [], []

    top3_correct = 0
    top5_correct = 0

    test_files = sorted([f for f in os.listdir(TEST_DIR) if is_png(f)])
    print(f"[INFO] Test images: {len(test_files)}")

    for fname in test_files:
        true_label = fname[:2]
        gray = load_gray_image(os.path.join(TEST_DIR, fname))

        probs = predict_page_probs(model, gray, num_classes)

        true_idx = class_to_idx[true_label]
        pred_idx = int(np.argmax(probs))

        top3 = np.argsort(probs)[-3:]
        top5 = np.argsort(probs)[-5:]

        if true_idx in top3:
            top3_correct += 1
        if true_idx in top5:
            top5_correct += 1

        y_true.append(true_label)
        y_pred.append(idx_to_class[pred_idx])
        y_true_idx.append(true_idx)
        y_score.append(probs)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    y_true_oh = keras.utils.to_categorical(y_true_idx, num_classes)
    y_score = np.vstack(y_score)

    auc = roc_auc_score(
        y_true_oh,
        y_score,
        average="macro",
        multi_class="ovr",
    )

    top3_acc = top3_correct / len(test_files)
    top5_acc = top5_correct / len(test_files)

    print(f"[RESULT] Top-1 Accuracy : {acc:.4f}")
    print(f"[RESULT] Top-3 Accuracy : {top3_acc:.4f}")
    print(f"[RESULT] Top-5 Accuracy : {top5_acc:.4f}")
    print(f"[RESULT] Macro F1       : {f1:.4f}")
    print(f"[RESULT] Macro AUC      : {auc:.4f}")

    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("filename,actual_label,predicted_label\n")
        for fname, yt, yp in zip(test_files, y_true, y_pred):
            f.write(f"{fname},{yt},{yp}\n")

    print(f"[INFO] Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
