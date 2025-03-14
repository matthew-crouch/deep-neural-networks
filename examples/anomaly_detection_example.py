"""Example of anomaly detection using LSTM Autoencoder."""

import torch

from src.anomaly_detection.create_dataset import (
    generate_anomaly_dataset,
)
from src.anomaly_detection.training_pipeline import TrainingPipeline
from src.llms.model_zoo.models import AutoEncoder, MistralModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    x, y = generate_anomaly_dataset(
        n_samples=10000, n_features=10, random_state=42, sequence_length=0
    )
    xval, yval = generate_anomaly_dataset(
        n_samples=10000, n_features=10, random_state=42, sequence_length=0
    )

    config = {
        "input_size": x.shape[1],
        "sequence_length": 0,
        "hidden_size": 128,
        "num_layers": 20,
        "output_size": 2,
        "dropout": 0.1,
        "learning_rate": 0.00001,
        "batch_size": x.shape[0],
        "num_epochs": 100000,
        "early_stopping": True,
        "rank": 1,
        "world_size": torch.cuda.device_count(),
        "quantise": {"enabled": True},
    }

    # model = LSTMClassifier(
    #     config["input_size"],
    #     config["hidden_size"],
    #     config["num_layers"],
    #     config["output_size"],
    #     config["dropout"],
    # )
    # model = AutoEncoder(
    #     config["input_size"],
    #     config["dropout"],
    # )
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = MistralModel()

    training_pipeline = TrainingPipeline(model=model, configuration=config)
    training_pipeline.run(train_data=(x, y), val_data=(xval, yval))
