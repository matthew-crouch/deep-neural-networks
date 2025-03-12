"""Example of anomaly detection using LSTM Autoencoder."""

import torch

from src.anomaly_detection.create_dataset import (
    generate_anomaly_dataset,
)  # ,  preprocess_api_dataset,
from src.anomaly_detection.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    x, y = generate_anomaly_dataset(n_samples=10000, n_features=10, random_state=42)
    xval, yval = generate_anomaly_dataset(n_samples=10000, n_features=10, random_state=420)

    config = {
        "input_size": x.shape[2],
        "sequence_length": x.shape[1],
        "hidden_size": 128,
        "num_layers": 20,
        "output_size": 2,
        "dropout": 0.1,
        "learning_rate": 0.0001,
        "batch_size": x.shape[0],
        "num_epochs": 100000,
        "early_stopping": True,
        "rank": 1,
        "world_size": torch.cuda.device_count(),
        "quantise": {"enabled": False, "num_bits": 8},
    }

    training_pipeline = TrainingPipeline(configuration=config)
    training_pipeline.run(train_data=(x, y), val_data=(xval, yval))
