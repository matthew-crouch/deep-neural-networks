"""Test the training pipeline."""

import pytest

from src.anomaly_detection.create_dataset import generate_anomaly_dataset
from src.anomaly_detection.model_io import ModelIo
from src.anomaly_detection.training_pipeline import TrainingPipeline
from src.llms.model_zoo.lstm import LSTMClassifier


def test_run_anomaly_detection():
    """Test the run_anomaly_detection function."""
    x, y = generate_anomaly_dataset(n_samples=100, n_features=10, random_state=42)
    xval, yval = generate_anomaly_dataset(n_samples=10000, n_features=10, random_state=420)
    config = {
        "input_size": x.shape[2],
        "sequence_length": x.shape[1],
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 2,
        "dropout": 0.25,
        "learning_rate": 0.001,
        "batch_size": x.shape[0],
        "num_epochs": 2,
        "early_stopping": True,
    }

    model = LSTMClassifier(
        config["input_size"],
        config["hidden_size"],
        config["num_layers"],
        config["output_size"],
        config["dropout"],
    )

    training_pipeline = TrainingPipeline(model=model, configuration=config)
    training_pipeline.run(train_data=(x, y), val_data=(xval, yval))


@pytest.mark.skip("Not implemented")
def test_load_model():
    """Test the load_model function."""
    model_io = ModelIo()
    model = model_io.load("./models/anomaly_detection.onnx")
    assert model is not None
