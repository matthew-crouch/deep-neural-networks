from src.run_anomaly_detection import TrainingPipeline, TrainingConfig
from src.create_dataset import generate_anomaly_dataset


def test_run_anomaly_detection():
    """Test the run_anomaly_detection function."""
    n_features = 13

    x, y = generate_anomaly_dataset(n_samples=15000, n_features=n_features, random_state=42)
    config = {
        "input_size": n_features,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 2,
        "dropout": 0.25,
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_epochs": 10,
    }
    breakpoint()
    training_pipeline = TrainingPipeline(configuration=config)
