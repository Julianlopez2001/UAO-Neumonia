import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.integrator import run_prediction_pipeline


def main() -> None:
    image_path = "data/raw/person1497_virus_2607.jpeg"
    model_path = "models/conv_MLP_84.h5"

    result = run_prediction_pipeline(image_path, model_path)

    print("\nRESULTADO DEL MODELO\n")
    print("Clase predicha:", result["label"])
    print("Probabilidad:", f"{result['probability']:.2f}%")
    print("Imagen usada:", image_path)


if __name__ == "__main__":
    main()