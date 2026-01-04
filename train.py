import os
import random
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from patching import PatchConfig, extract_line_patches_from_gray
from utils import (
    ensure_dir,
    extract_writer_label,
    is_png,
    load_gray_image,
    save_class_map,
)

# Paths
TRAIN_DIR = "dataset/train"
PROCESSED_TRAIN_DIR = "processed/train"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "writer_model.keras")
CLASS_MAP_PATH = "class_map.json"

# Patch configuration (must match run.py)
CFG = PatchConfig(
    patch_h=128,
    patch_w=256,
    seg_per_line=8,
    min_ink_ratio=0.01,
)

IMG_H, IMG_W = CFG.patch_h, CFG.patch_w
BATCH_SIZE = 32
EPOCHS = 45
SEED = 42
MAX_PATCHES_PER_WRITER = 400


# Reproducibility
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Patch generation + saving (turn one page per writer into many training samples)
def save_patches_for_training() -> List[str]:
    ensure_dir(PROCESSED_TRAIN_DIR)

    train_files = sorted([f for f in os.listdir(TRAIN_DIR) if is_png(f)])
    labels = sorted({extract_writer_label(f) for f in train_files})

    for lab in labels:
        ensure_dir(os.path.join(PROCESSED_TRAIN_DIR, lab))

    writer_to_patches = {lab: [] for lab in labels}

    for fname in train_files:
        label = extract_writer_label(fname)
        gray = load_gray_image(os.path.join(TRAIN_DIR, fname))
        patches = extract_line_patches_from_gray(gray, CFG)

        base = os.path.splitext(fname)[0]
        for i, p in enumerate(patches):
            writer_to_patches[label].append((base, i, p))

    rng = random.Random(SEED)
    total_saved = 0

    for lab, items in writer_to_patches.items():
        if len(items) > MAX_PATCHES_PER_WRITER:
            items = rng.sample(items, MAX_PATCHES_PER_WRITER)

        out_dir = os.path.join(PROCESSED_TRAIN_DIR, lab)
        for base, i, p in items:
            out_name = f"{lab}_{base}_P{i:04d}.png"
            out_path = os.path.join(out_dir, out_name)
            import cv2

            cv2.imwrite(out_path, p)
            total_saved += 1

    print(f"[INFO] Saved train patches: {total_saved}")
    print(f"[INFO] Writers/classes: {len(labels)}")
    return labels


# CNN architecture (trained from scratch, deeper + regularized)
def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(IMG_H, IMG_W, 1))

    def block(x, f):
        x = layers.Conv2D(
            f,
            3,
            padding="same",
            kernel_initializer="he_uniform",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
        return x

    x = block(inputs, 32)
    x = block(x, 64)
    x = block(x, 128)
    x = block(x, 256)
    x = block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        512,
        activation="relu",
        kernel_initializer="he_uniform",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_initializer="he_uniform",
    )(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4,
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# tf.data pipeline (load patches, normalize, light augmentation)
def make_dataset(paths, labels, num_classes, training):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = 1.0 - img
        img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)

        if training:
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)

        y = tf.one_hot(label, num_classes)
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(6000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Training loop (split, train, callbacks, save artifacts)
def main():
    set_seeds(SEED)

    labels = save_patches_for_training()
    class_to_idx = {lab: i for i, lab in enumerate(labels)}
    save_class_map(CLASS_MAP_PATH, class_to_idx)
    num_classes = len(labels)

    all_paths, all_y = [], []
    for lab in labels:
        d = os.path.join(PROCESSED_TRAIN_DIR, lab)
        for f in os.listdir(d):
            if is_png(f):
                all_paths.append(os.path.join(d, f))
                all_y.append(class_to_idx[lab])

    all_paths = np.array(all_paths)
    all_y = np.array(all_y)

    print(f"[INFO] Total patches after cap: {len(all_paths)}")

    X_train, X_val, y_train, y_val = train_test_split(
        all_paths,
        all_y,
        test_size=0.2,
        random_state=SEED,
        stratify=all_y,
    )

    train_ds = make_dataset(X_train, y_train, num_classes, True)
    val_ds = make_dataset(X_val, y_val, num_classes, False)

    model = build_model(num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    ensure_dir(MODEL_DIR)
    model.save(MODEL_PATH)
    print(f"[INFO] Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
