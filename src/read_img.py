"""
Module responsible for reading medical images.

Supports DICOM and common image formats such as JPG, JPEG and PNG.
"""

from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image


def read_image(path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Read an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        tuple:
            numpy array for processing
            PIL image for visualization
    """

    extension = Path(path).suffix.lower()

    # DICOM IMAGE
    if extension == ".dcm":
        dicom_file = pydicom.dcmread(path)
        image_array = dicom_file.pixel_array.astype("float32")

        image_array = (np.maximum(image_array, 0) / image_array.max()) * 255
        image_array = np.uint8(image_array)

        display_image = Image.fromarray(image_array)

        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        return image_rgb, display_image

    # NORMAL IMAGE
    if extension in [".jpg", ".jpeg", ".png"]:
        image_bgr = cv2.imread(path)

        if image_bgr is None:
            raise ValueError(f"Cannot read image {path}")

        display_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        image_array = image_bgr.astype("float32")
        image_array = (np.maximum(image_array, 0) / image_array.max()) * 255
        image_array = np.uint8(image_array)

        return image_array, display_image

    raise ValueError(f"Unsupported format: {extension}")