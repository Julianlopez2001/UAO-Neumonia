"""
Image preprocessing module.

Transforms input images before model inference.
"""

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for CNN model.

    Steps
    -----
    1. Resize to 512x512
    2. Convert to grayscale
    3. Apply CLAHE contrast enhancement
    4. Normalize
    5. Add batch dimension

    Args:
        image: numpy image array

    Returns:
        preprocessed image tensor
    """

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = clahe.apply(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image