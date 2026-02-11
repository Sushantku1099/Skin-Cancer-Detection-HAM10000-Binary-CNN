#!/usr/bin/env python3
"""Train a binary skin-cancer classifier on HAM10000 using DenseNet201."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


def set_global_seed(seed: int) -> None:
    """Ensure reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Determinism can fail on some hardware/drivers; training will still run.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="HAM10000 root directory")
    parser.add_argument("--metadata", type=str, default="HAM10000_metadata.csv", help="Metadata CSV filename")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--learning-rate-stage1", type=float, default=1e-3)
    parser.add_argument("--learning-rate-stage2", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_prepare_dataframe(data_dir: Path, metadata_name: str) -> pd.DataFrame:
    metadata_path = data_dir / metadata_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Binary mapping requested by specification:
    # Benign = nv, bkl, df, vasc
    # Malignant = mel, bcc, akiec
    binary_map: Dict[str, int] = {
        "nv": 0,
        "bkl": 0,
        "df": 0,
        "vasc": 0,
        "mel": 1,
        "bcc": 1,
        "akiec": 1,
    }

    # Keep this commented section for future 7-class training workflows.
    # seven_class_map = {
    #     "akiec": 0,
    #     "bcc": 1,
    #     "bkl": 2,
    #     "df": 3,
    #     "mel": 4,
    #     "nv": 5,
    #     "vasc": 6,
    # }

    df = df[df["dx"].isin(binary_map)].copy()
    df["label"] = df["dx"].map(binary_map).astype("int32")

    candidate_dirs = [
        data_dir / "HAM10000_images",
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        data_dir / "images",
    ]

    def resolve_image_path(image_id: str) -> str:
        for img_dir in candidate_dirs:
            image_path = img_dir / f"{image_id}.jpg"
            if image_path.exists():
                return str(image_path)
        raise FileNotFoundError(
            f"Image {image_id}.jpg not found in any known HAM10000 image directories."
        )

    df["image_path"] = df["image_id"].map(resolve_image_path)
    return df


def decode_and_preprocess(path: tf.Tensor, label: tf.Tensor, img_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.densenet.preprocess_input(image)
    label = tf.cast(label, tf.float32)
    return image, label


def build_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: decode_and_preprocess(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int, lr: float) -> Tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.2),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    base_model = keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    model = keras.Model(inputs, outputs, name="binary_dermnet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model, base_model


def build_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(
            "binary_dermnet_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    df = load_and_prepare_dataframe(args.data_dir, args.metadata)

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=df["label"],
    )

    train_paths = train_df["image_path"].to_numpy()
    val_paths = val_df["image_path"].to_numpy()
    train_labels = train_df["label"].to_numpy().astype("float32")
    val_labels = val_df["label"].to_numpy().astype("float32")

    train_ds = build_dataset(train_paths, train_labels, args.img_size, args.batch_size, training=True, seed=args.seed)
    val_ds = build_dataset(val_paths, val_labels, args.img_size, args.batch_size, training=False, seed=args.seed)

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels.astype("int32")),
        y=train_labels.astype("int32"),
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}

    model, base_model = build_model(args.img_size, args.learning_rate_stage1)
    callbacks = build_callbacks()

    print("\nStage 1/2: Training with DenseNet201 frozen")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nStage 2/2: Fine-tuning last 40 DenseNet201 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    for layer in base_model.layers[-40:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate_stage2),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1 + args.epochs_stage2,
        initial_epoch=args.epochs_stage1,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = keras.models.load_model("binary_dermnet_model.keras")
    eval_results = best_model.evaluate(val_ds, verbose=0, return_dict=True)
    val_auc = float(eval_results["auc"])

    y_prob = best_model.predict(val_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype("int32")
    cm = confusion_matrix(val_labels.astype("int32"), y_pred)

    print("\nFinal validation AUC:", f"{val_auc:.6f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
