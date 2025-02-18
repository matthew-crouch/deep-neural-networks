"""LSTM model for anomaly detection."""

import torch
from pydantic import BaseModel
from torch import nn
import torch.onnx
import onnx
import pandas as pd
from src.create_dataset import generate_anomaly_dataset


class LSTMClassifier(nn.Module):
    """LSTM model for anomaly detection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.25,
    ):
        """Initialize the LSTM model."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, output_size)

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass Function."""
        h_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward pass through LSTM layer, output shape: (batch_size, seq_length, hidden_size)
        output, _ = self.lstm(x, (h_init, c_init))

        return self.fully_connected(output[:, -1, :])


class TrainingConfig(BaseModel):
    """Dataclass for training configuration."""

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int


class TrainingPipeline:
    """Training pipeline for the Anomaly Detection model."""

    def __init__(self, configuration: dict):
        """Initialize the training pipeline."""
        self.configuration = TrainingConfig(**configuration)
        self.model = LSTMClassifier(
            self.configuration.input_size,
            self.configuration.hidden_size,
            self.configuration.num_layers,
            self.configuration.output_size,
            self.configuration.dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configuration.learning_rate
        )

    def _model_packaging(self, filename: str):
        """Package the model after training to onnx."""
        # torch.onnx.export(self.model, )
        return None

    def run(self, data: pd.DataFrame):
        """Run the training Pipeline"""
