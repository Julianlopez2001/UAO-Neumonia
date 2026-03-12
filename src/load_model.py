"""
Module responsible for loading the trained pneumonia classification model.
"""

from functools import lru_cache

import tensorflow as tf

# Force TensorFlow to use CPU only.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


@lru_cache(maxsize=1)
def load_trained_model(model_path: str):
    """
    Load and cache the trained Keras model.

    Args:
        model_path: Path to the .h5 model file.

    Returns:
        Loaded TensorFlow/Keras model.
    """
    return tf.keras.models.load_model(model_path, compile=False)