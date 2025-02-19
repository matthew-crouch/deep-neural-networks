"""Test the training pipeline."""

from src.anomaly_detection import TrainingPipeline
from src.create_dataset import generate_anomaly_dataset


def test_run_anomaly_detection():
    """Test the run_anomaly_detection function."""
    x, y = generate_anomaly_dataset(n_samples=100, n_features=10, random_state=42)
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
    }

    training_pipeline = TrainingPipeline(configuration=config)
    training_pipeline.run(train_data=(x, y))
