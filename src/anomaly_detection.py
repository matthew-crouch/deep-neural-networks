"""LSTM model for anomaly detection."""

import logging
import uuid
from pathlib import Path

import torch
import torch.onnx
from pydantic import BaseModel
from torch import nn
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass Function."""
        h_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # forward pass through LSTM layer, output shape: (batch_size, seq_length, hidden_size)
        output, _ = self.lstm(x.to(self.device), (h_init, c_init))

        return self.fully_connected(output[:, -1, :])


class TrainingConfig(BaseModel):
    """Dataclass for training configuration."""

    input_size: int
    hidden_size: int
    sequence_length: int
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
        self.run_id = Path(str(uuid.uuid4()))
        Path.mkdir(self.run_id, parents=True, exist_ok=True)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_model_package(self, filename: str) -> None:
        """Package the model after training to onnx.

        :param filename: Name of the file to save the model.
        """
        self.model.eval()

        if not os.path.exists("models"):
            os.makedirs("models")

        torch.onnx.export(
            self.model,
            torch.randn(1, self.configuration.sequence_length, self.configuration.input_size),
            filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            opset_version=11,
        )

    def model_checkpoint(self):
        """Save the model checkpoint."""
        torch.save(self.model.state_dict(), f"{self.run_id}/model_{uuid.uuid4()}.pth")

    def make_dataloader(self, x: torch.Tensor, y: torch.Tensor):
        """Create a DataLoader from the input data.

        :param x: Input data.
        :param y: Target data.
        :return: DataLoader for training data.
        """
        train_data = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=self.configuration.batch_size, shuffle=True
        )
        return train_loader

    def evaluate(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate the model."""
        raise NotImplementedError

    def train(self, train_loader: torch.utils.data.DataLoader):
        """Train the model.

        :param train_loader: DataLoader for training data.
        """
        logger.info("Training Started...")
        self.model.train()
        for epoch in range(self.configuration.num_epochs):
            for _i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                # Forward Pass
                outputs = self.model(data)

                ## We only select the last timestep (not generalised for all solutions)
                # This will only predict one label per sequence, not one per timestep.
                loss = self.criterion(outputs, labels[:, -1])

                # Backward Pass and Optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logger.info(f"Epoch: {epoch}, Loss: {loss.item()}")
                self.model_checkpoint()
        logger.info("Training Finished...")

    def run(self, x: torch.Tensor, y: torch.Tensor):
        """Run the training Pipeline.

        :param x: Input data.
        :param y: Target data.
        """
        train_loader = self.make_dataloader(x, y)
        self.train(train_loader)

        self.create_model_package(filename="./models/anomaly_detection.onnx")
