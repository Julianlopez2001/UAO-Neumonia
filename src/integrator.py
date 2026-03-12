"""
Main integration pipeline for pneumonia classification.
"""

import numpy as np

from src.grad_cam import generate_grad_cam
from src.load_model import load_trained_model
from src.preprocess_img import preprocess_image
from src.read_img import read_image


CLASS_LABELS = {
    0: "bacteriana",
    1: "normal",
    2: "viral",
}


def run_prediction_pipeline(image_path: str, model_path: str) -> dict:
    """
    Run the full prediction pipeline.

    Args:
        image_path: Path to the input image.
        model_path: Path to the trained model.

    Returns:
        Dictionary containing:
            - label: predicted class label
            - probability: prediction confidence in percent
            - display_image: PIL image for UI display
            - heatmap: Grad-CAM image
    """
    image_array, display_image = read_image(image_path)
    preprocessed_image = preprocess_image(image_array)

    model = load_trained_model(model_path)
    predictions = model.predict(preprocessed_image, verbose=0)

    predicted_index = int(np.argmax(predictions))
    probability = float(np.max(predictions) * 100)
    label = CLASS_LABELS.get(predicted_index, "desconocida")

    heatmap = generate_grad_cam(image_array, model)

    return {
        "label": label,
        "probability": probability,
        "display_image": display_image,
        "heatmap": heatmap,
    }