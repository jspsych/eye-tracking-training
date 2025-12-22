# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains machine learning models for gaze prediction (eye tracking) using deep learning. The project trains models on eye images with associated gaze coordinates to predict where a user is looking on a screen based on webcam images of their eyes.

## Project Structure

- **models/** - Training notebooks for the core gaze prediction models
  - `densenet-regression-mask-trick-model.ipynb` - Main training notebook using DenseNet backbone with masked calibration/target approach

- **analysis/** - Evaluation and analysis notebooks
  - `analysis-of-individual-model-run.ipynb` - Evaluates trained models on test data
  - `training-set-size-analysis.ipynb` - Analyzes impact of training set size on performance
  - `mpiigaze-evaluation.ipynb` - Evaluates models on MPIIGaze dataset
  - `webgazer-comparison.ipynb` - Compares performance against WebGazer
  - `single_eye_tfrecords/` - TFRecord files containing eye images, landmarks, and gaze coordinates
  - `full_model.weights.h5` - Saved model weights

- **et_util/** - Custom utility modules for model training and evaluation
  - `custom_layers.py` - Custom Keras layers (SimpleTimeDistributed, MaskedWeightedRidgeRegressionLayer, MaskInspectorLayer)
  - `custom_loss.py` - Custom loss functions (normalized_weighted_euc_dist)
  - `dataset_utils.py` - Dataset loading, processing, and parsing utilities (process_tfr_to_tfds, parse_single_eye_tfrecord, rescale_coords_map)
  - `model_analysis.py` - Model evaluation and visualization tools (plot_model_performance)

## Key Architecture Concepts

### Model Architecture Overview

The model uses a **calibration-based approach** with masking:

1. **Input Format**: Each training example contains up to 144 images per subject with associated gaze coordinates
2. **Calibration vs Target Separation**: Images are split into:
   - Calibration images (8-40 images): Used to learn subject-specific gaze patterns via ridge regression
   - Target images: Remaining images used for prediction and evaluation
3. **Masking System**: Two masks distinguish calibration from target images:
   - `Input_Calibration_Mask`: Explicit mask marking calibration images
   - Target mask: Derived mask used as sample weights in loss function

### Key Components

- **Backbone**: DenseNet (configurable) extracts features from 36x144x1 grayscale eye images
- **Embedding**: Dense layer produces fixed-size embedding (default 200-dim) from backbone output
- **Calibration Weights**: Learned importance weights for each calibration point
- **Ridge Regression Layer** (`MaskedWeightedRidgeRegressionLayer`): Learns subject-specific mapping from embeddings to gaze coordinates using weighted ridge regression with calibration mask
- **Output**: Predicted (x,y) coordinates normalized to [0,1] range

### Custom Loss Function

`normalized_weighted_euc_dist`: Euclidean distance weighted by 1.778 on x-axis (16:9 aspect ratio correction), normalized to diagonal distance for interpretability (max distance = 100).

## Environment Setup

This project is designed to run in **Google Colab** and uses:
- TensorFlow/Keras with Keras 3.x API
- Weights & Biases (wandb) for experiment tracking
- OSF (Open Science Framework) for dataset storage
- Custom utilities from the `et_util` package (included in this repository)

### Key Dependencies
```python
# Installed in notebook cells
osfclient
wandb
keras
keras-hub
```

### Environment Variables (stored in analysis/.env)
- `WANDB_API_KEY` - Weights & Biases API key
- `OSF_TOKEN` - Open Science Framework token for dataset access
- `OSF_USERNAME` - OSF username

## Working with the Codebase

### Running Model Training

The main training notebook is `models/densenet-regression-mask-trick-model.ipynb`. It:

1. Downloads training data from OSF (`single_eye_tfrecords.tar.gz`)
2. Processes TFRecords into TF Datasets with subject grouping
3. Applies masking to separate calibration/target images
4. Trains the model with augmentation
5. Evaluates on test set with fixed calibration grid
6. Logs results to Weights & Biases

**Note**: The notebook expects Google Colab environment with access to `userdata` for API keys.

### Data Format

TFRecords contain:
- `eye_img`: 36x144 grayscale eye image (uint8, stored as string)
- `landmarks`: MediaPipe face mesh landmarks (478 points x 3 coords)
- `x`, `y`: Gaze coordinates (float32, range 0-100)
- `subject_id`: Integer identifier for grouping
- `img_width`, `img_height`: Original image dimensions

### Augmentation Pipeline

Two-stage augmentation (controlled by `AUGMENTATION` flag):

1. **Non-affine augmentations** (can be per-image or per-sequence):
   - Gaussian blur (probability-based)
   - Brightness, contrast, gamma adjustments
   - Gaussian noise

2. **Affine augmentations** (always consistent across sequence):
   - Horizontal flip
   - Rotation (±5% * π radians)
   - Zoom (0.9-1.1x)
   - Translation (±10% of image size)
   - Perspective transform

### Key Configuration Parameters

Located in the training notebook:
- `EMBEDDING_DIM`: 200 - Size of embedding vector
- `RIDGE_REGULARIZATION`: 0.1 - Ridge regression lambda
- `MIN_CAL_POINTS`: 8 - Minimum calibration images
- `MAX_CAL_POINTS`: 40 - Maximum calibration images
- `BACKBONE`: "densenet" - Feature extraction architecture
- `BATCH_SIZE`: 5 - Subjects per training batch
- `TRAIN_EPOCHS`: 50 - Training duration
- `MAX_TARGETS`: 144 - Maximum images per subject (for padding)

### Model Evaluation

Standard evaluation uses a fixed 20-point calibration grid (5x4 grid covering the screen). The `analysis-of-individual-model-run.ipynb` notebook:
- Loads trained model weights
- Evaluates on held-out test subjects
- Computes per-subject loss distributions
- Generates visualizations and logs to wandb

### Custom Layers

- `SimpleTimeDistributed`: Applies a layer to each timestep without for-loops (uses reshape)
- `MaskedWeightedRidgeRegressionLayer`: Core regression layer with explicit calibration masking and learned per-point weights

## Common Tasks

### Training a new model
Open `models/densenet-regression-mask-trick-model.ipynb` in Colab, ensure credentials are set, and run all cells.

### Analyzing model performance
Use `analysis/analysis-of-individual-model-run.ipynb` to load saved weights and evaluate on test data.

### Modifying the backbone
Change `BACKBONE` parameter and add corresponding backbone creation function in the "Backbones" section (RedNet, Involution, EfficientNet examples provided).

### Adjusting calibration points
Modify `MIN_CAL_POINTS`/`MAX_CAL_POINTS` for random calibration or pass fixed `calibration_points` tensor to `prepare_masked_dataset()` for deterministic evaluation.
