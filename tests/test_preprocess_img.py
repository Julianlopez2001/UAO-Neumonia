"""
Unit tests for image preprocessing.
"""

import numpy as np

from src.preprocess_img import preprocess_image


def test_preprocess_image_shape() -> None:
    """
    Verify that the preprocessed image has the expected shape.
    """
    image = np.random.randint(0, 255, (700, 700, 3), dtype=np.uint8)

    result = preprocess_image(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_image_range() -> None:
    """
    Verify that the preprocessed image is normalized between 0 and 1.
    """
    image = np.random.randint(0, 255, (700, 700, 3), dtype=np.uint8)

    result = preprocess_image(image)

    assert result.min() >= 0.0
    assert result.max() <= 1.0