# Skin-Cancer-Detection-HAM10000-Binary-CNN

TensorFlow training pipeline for a **binary skin cancer classifier** on the HAM10000 dataset using **DenseNet201**.

## What this implements

- DenseNet201 backbone (`weights="imagenet"`, `include_top=False`)
- Binary label conversion from HAM10000 classes:
  - **Benign (0):** `nv`, `bkl`, `df`, `vasc`
  - **Malignant (1):** `mel`, `bcc`, `akiec`
- `tf.data` input pipeline
- Image size: **224x224**
- Data augmentation:
  - `RandomFlip`
  - `RandomRotation`
  - `RandomZoom`
- `class_weight` handling for imbalance
- `BinaryCrossentropy` loss
- Metrics:
  - Accuracy
  - AUC
  - Precision
  - Recall
- Two-stage training:
  1. Freeze DenseNet201 base
  2. Unfreeze the last 40 layers for fine-tuning
- Callbacks:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint` saving best model to `binary_dermnet_model.keras`
- Reproducibility via random seeds
- Final reporting of validation AUC and confusion matrix
- Commented section retained for future 7-class classification mapping

## Expected dataset layout

Point `--data-dir` to a folder containing `HAM10000_metadata.csv` and one or more of:

- `HAM10000_images/`
- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`
- `images/`

## Usage

```bash
python train_binary_dermnet.py \
  --data-dir /path/to/HAM10000 \
  --batch-size 32 \
  --epochs-stage1 10 \
  --epochs-stage2 20 \
  --seed 42
```

At the end of training, the script prints:

- Final validation AUC
- Confusion matrix (`rows=true labels`, `cols=predicted labels`)

and saves best model as:

- `binary_dermnet_model.keras`
