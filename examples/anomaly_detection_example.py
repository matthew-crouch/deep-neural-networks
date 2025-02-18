"""Example of anomaly detection using LSTM Autoencoder."""

from src.anomaly_detection import TrainingPipeline
from src.create_dataset import generate_anomaly_dataset

if __name__ == "__main__":
    x, y = generate_anomaly_dataset(n_samples=10000, n_features=10, random_state=42)
    config = {
        "input_size": x.shape[2],
        "sequence_length": x.shape[1],
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 2,
        "dropout": 0.25,
        "learning_rate": 0.001,
        "batch_size": x.shape[0],
        "num_epochs": 100,
    }

    training_pipeline = TrainingPipeline(configuration=config)
    training_pipeline.run(x=x, y=y)
