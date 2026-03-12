"""
Unit tests for the integration pipeline.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.integrator import run_prediction_pipeline


@patch("src.integrator.generate_grad_cam")
@patch("src.integrator.load_trained_model")
@patch("src.integrator.read_image")
def test_run_prediction_pipeline_returns_expected_keys(
    mock_read_image,
    mock_load_trained_model,
    mock_generate_grad_cam,
) -> None:
    """
    Verify that the prediction pipeline returns the expected dictionary keys.
    """
    mock_read_image.return_value = (
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
        "display_image_mock",
    )

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.8, 0.1, 0.1]])
    mock_load_trained_model.return_value = fake_model
    mock_generate_grad_cam.return_value = np.zeros((512, 512), dtype=np.uint8)

    result = run_prediction_pipeline("fake_image.dcm", "models/MLP_84.h5")

    assert "label" in result
    assert "probability" in result
    assert "display_image" in result
    assert "heatmap" in result
    assert result["label"] == "bacteriana"