"""
Script para verificar que el modelo conv_MLP_84.h5 carga correctamente
y mostrar su arquitectura.
"""

import tensorflow as tf

MODEL_PATH = "models/conv_MLP_84.h5"


def main():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("\nModelo cargado correctamente.\n")
    model.summary()

    print("\nCapas del modelo:\n")
    for layer in model.layers:
        print(layer.name, "-", layer.__class__.__name__)


if __name__ == "__main__":
    main()