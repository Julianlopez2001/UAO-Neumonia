"""
Module responsible for generating Grad-CAM visual explanations.
"""

import cv2
import numpy as np
import tensorflow as tf

from src.preprocess_img import preprocess_image


def generate_grad_cam(
    image: np.ndarray,
    model,
    layer_name: str = "conv10_thisone",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap superimposed on the original image.

    Args:
        image: Original image as NumPy array.
        model: Loaded Keras model.
        layer_name: Name of the convolutional layer used for Grad-CAM.

    Returns:
        Image with the Grad-CAM heatmap superimposed.
    """
    processed_image = preprocess_image(image)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([processed_image], training=False)

        # Si predictions viene como lista, tomar el primer elemento
        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]

    gradients = tape.gradient(class_channel, conv_outputs)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_gradients, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    resized_original = cv2.resize(image, (512, 512))
    superimposed = cv2.addWeighted(resized_original, 0.8, heatmap, 0.4, 0)

    return superimposed[:, :, ::-1]