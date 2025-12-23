"""
Inference utilities for flexible gaze prediction models.

This module provides utilities for creating inference models that can accept
arbitrary numbers of points (unlike the training model which requires a fixed
number). This is achieved by using dynamic shapes and disabling JIT compilation.
"""

import keras
import numpy as np
from typing import Dict, Optional, List

from et_util.custom_layers import (
    SimpleTimeDistributed,
    MaskedWeightedRidgeRegressionLayer,
    ResidualBlock,
    AddPositionalEmbedding,
)
from et_util.custom_loss import normalized_weighted_euc_dist


def _create_embedding_model(
    embedding_dim: int = 200,
    densenet_stackwise_num_repeats: List[int] = [4, 4, 4],
) -> keras.Model:
    """
    Create the embedding model (backbone + embedding layer).

    Args:
        embedding_dim: Size of the embedding vector
        densenet_stackwise_num_repeats: DenseNet configuration

    Returns:
        Keras Model that maps eye images to embeddings
    """
    import keras_hub

    image_shape = (36, 144, 1)
    input_eyes = keras.layers.Input(shape=image_shape)

    # Rescale to [0, 1]
    eyes_rescaled = keras.layers.Rescaling(scale=1./255)(input_eyes)

    # Create DenseNet backbone
    backbone = keras_hub.models.DenseNetBackbone(
        stackwise_num_repeats=densenet_stackwise_num_repeats,
        image_shape=(36, 144, 1),
    )

    backbone_encoder = backbone(eyes_rescaled)
    flatten_compress = keras.layers.Flatten()(backbone_encoder)
    eye_embedding = keras.layers.Dense(units=embedding_dim, activation="tanh")(flatten_compress)

    embedding_model = keras.Model(
        inputs=input_eyes,
        outputs=eye_embedding,
        name="Eye_Image_Embedding"
    )

    return embedding_model


def _create_flexible_model_architecture(
    embedding_dim: int = 200,
    ridge_regularization: float = 0.1,
    densenet_stackwise_num_repeats: List[int] = [4, 4, 4],
) -> keras.Model:
    """
    Create the full model architecture with flexible (dynamic) shapes.

    Unlike the training model which uses fixed shapes (e.g., 144 points),
    this model uses None for the time dimension to accept any number of points.

    Args:
        embedding_dim: Size of the embedding vector
        ridge_regularization: Ridge regression lambda parameter
        densenet_stackwise_num_repeats: DenseNet configuration

    Returns:
        Keras Model with flexible input shapes
    """
    # Create embedding model
    embedding_model = _create_embedding_model(
        embedding_dim=embedding_dim,
        densenet_stackwise_num_repeats=densenet_stackwise_num_repeats,
    )

    # Define flexible inputs (None instead of fixed number)
    input_all_images = keras.layers.Input(
        shape=(None, 36, 144, 1),  # Flexible time dimension
        name="Input_All_Images"
    )

    input_all_coords = keras.layers.Input(
        shape=(None, 2),  # Flexible time dimension
        name="Input_All_Coords"
    )

    input_cal_mask = keras.layers.Input(
        shape=(None,),  # Flexible time dimension
        name="Input_Calibration_Mask",
    )

    # Apply the embedding model to all images
    all_embeddings = SimpleTimeDistributed(
        embedding_model,
        name="Image_Embeddings"
    )(input_all_images)

    # Calculate importance weights for calibration points
    calibration_weights = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="Calibration_Weights"
    )(all_embeddings)

    # Ridge regression layer
    ridge = MaskedWeightedRidgeRegressionLayer(
        ridge_regularization,
        name="Regression"
    )(
        [
            all_embeddings,
            input_all_coords,
            calibration_weights,
            input_cal_mask,
        ],
    )

    # Create the full model
    full_model = keras.Model(
        inputs=[
            input_all_images,
            input_all_coords,
            input_cal_mask,
        ],
        outputs=ridge,
        name="FlexibleGazeModel"
    )

    return full_model


def create_flexible_inference_model(
    trained_model_path: str,
) -> keras.Model:
    """
    Create a flexible inference model that accepts arbitrary point counts.

    This function loads a trained model and rewires it with new Input layers
    that have flexible shapes. All trained layers and weights are preserved.

    Args:
        trained_model_path: Path to the saved .keras model file

    Returns:
        A Keras model with flexible input shapes (None instead of fixed point counts)

    Example:
        >>> model = create_flexible_inference_model('full_model.keras')
        >>> # Use with any number of points
        >>> predictions = model.predict({
        ...     "Input_All_Images": images,      # (1, n_points, 36, 144, 1)
        ...     "Input_All_Coords": coords,      # (1, n_points, 2)
        ...     "Input_Calibration_Mask": mask   # (1, n_points)
        ... })
    """
    # Custom objects needed to load the trained model
    custom_objects = {
        "SimpleTimeDistributed": SimpleTimeDistributed,
        "MaskedWeightedRidgeRegressionLayer": MaskedWeightedRidgeRegressionLayer,
        "normalized_weighted_euc_dist": normalized_weighted_euc_dist,
        "ResidualBlock": ResidualBlock,
        "AddPositionalEmbedding": AddPositionalEmbedding,
    }

    # Load the trained model
    trained_model = keras.models.load_model(
        trained_model_path,
        custom_objects=custom_objects,
    )

    # Extract existing trained layers (they keep their weights)
    image_embeddings_layer = trained_model.get_layer("Image_Embeddings")
    calibration_weights_layer = trained_model.get_layer("Calibration_Weights")
    regression_layer = trained_model.get_layer("Regression")

    # Create new flexible inputs (None instead of fixed number like 144)
    input_all_images = keras.layers.Input(
        shape=(None, 36, 144, 1),
        name="Input_All_Images"
    )
    input_all_coords = keras.layers.Input(
        shape=(None, 2),
        name="Input_All_Coords"
    )
    input_cal_mask = keras.layers.Input(
        shape=(None,),
        name="Input_Calibration_Mask",
    )

    # Wire existing layers to new flexible inputs
    all_embeddings = image_embeddings_layer(input_all_images)
    calibration_weights = calibration_weights_layer(all_embeddings)
    output = regression_layer([
        all_embeddings,
        input_all_coords,
        calibration_weights,
        input_cal_mask,
    ])

    # Create new model with flexible inputs
    flexible_model = keras.Model(
        inputs=[input_all_images, input_all_coords, input_cal_mask],
        outputs=output,
        name="FlexibleGazeModel"
    )

    # Compile without JIT for dynamic shape support
    flexible_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=normalized_weighted_euc_dist,
        jit_compile=False,
    )

    return flexible_model


def prepare_flexible_inference_inputs(
    images: np.ndarray,
    coords: np.ndarray,
    cal_mask: np.ndarray,
    validate_shapes: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Prepare inputs for the flexible inference model.

    Unlike training data preparation, this does NOT pad to a fixed size
    or use tf.ensure_shape(). It accepts data with any number of points.

    Args:
        images: Array of shape (n_points, 36, 144, 1) - eye images
        coords: Array of shape (n_points, 2) - gaze coordinates
        cal_mask: Array of shape (n_points,) - calibration mask (1 for calibration, 0 for target)
        validate_shapes: Whether to validate input dimensions

    Returns:
        Dictionary with keys matching model input names, ready for model.predict()

    Raises:
        ValueError: If shapes are invalid

    Example:
        >>> inputs = prepare_flexible_inference_inputs(images, coords, cal_mask)
        >>> predictions = model.predict(inputs)
    """
    if validate_shapes:
        # Validate first dimension matches
        n_points = images.shape[0]
        if coords.shape[0] != n_points:
            raise ValueError(
                f"Number of points mismatch: images has {n_points}, "
                f"coords has {coords.shape[0]}"
            )
        if cal_mask.shape[0] != n_points:
            raise ValueError(
                f"Number of points mismatch: images has {n_points}, "
                f"cal_mask has {cal_mask.shape[0]}"
            )

        # Validate image shape
        if len(images.shape) != 4 or images.shape[1:] != (36, 144, 1):
            raise ValueError(
                f"Invalid image shape: expected (n, 36, 144, 1), got {images.shape}"
            )

        # Validate coords shape
        if len(coords.shape) != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Invalid coords shape: expected (n, 2), got {coords.shape}"
            )

        # Validate cal_mask shape
        if len(cal_mask.shape) != 1:
            raise ValueError(
                f"Invalid cal_mask shape: expected (n,), got {cal_mask.shape}"
            )

    # Add batch dimension and return as dictionary
    return {
        "Input_All_Images": np.expand_dims(images, axis=0),
        "Input_All_Coords": np.expand_dims(coords, axis=0),
        "Input_Calibration_Mask": np.expand_dims(cal_mask, axis=0),
    }


def predict_with_flexible_model(
    model: keras.Model,
    images: np.ndarray,
    coords: np.ndarray,
    cal_mask: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions using the flexible inference model.

    This is a convenience function that handles input preparation and
    prediction extraction in a single call.

    Args:
        model: Flexible inference model created by create_flexible_inference_model()
        images: Images array (n_points, 36, 144, 1)
        coords: Coordinates array (n_points, 2)
        cal_mask: Calibration mask (n_points,) - 1 for calibration points, 0 for targets

    Returns:
        Predicted coordinates array (n_points, 2)

    Example:
        >>> model = create_flexible_inference_model('full_model.keras')
        >>> predictions = predict_with_flexible_model(model, images, coords, cal_mask)
        >>> # predictions shape: (n_points, 2)
    """
    # Prepare inputs
    inputs = prepare_flexible_inference_inputs(images, coords, cal_mask)

    # Generate predictions
    predictions = model.predict(inputs, verbose=0)

    # Remove batch dimension
    return predictions[0]
