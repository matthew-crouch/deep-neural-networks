"""LSTM model for anomaly detection."""

import logging

import torch
import torch.onnx
from pydantic import BaseModel
from torch import nn

from src.model_io import ModelIo

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


class EarlyStopping:
    """Early stopping."""

    def __init__(self, tolerance: int = 5):
        """Initialize the early stopping object."""
        self.tolerance = tolerance
        self.counter = 0
        self.min_delta = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss):
        """Check if early stopping condition is met."""
        if val_loss > train_loss:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True

        return self.early_stop


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
    early_stopping: bool


class TrainingPipeline:
    """Training pipeline for the Anomaly Detection model."""

    def __init__(self, configuration: dict):
        """Initialize the training pipeline."""
        self.configuration = TrainingConfig(**configuration)

        self.early_stopping = EarlyStopping()
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

        self.model_io = ModelIo(self.model)

    def make_dataloader(self, data: tuple[torch.Tensor, torch.Tensor]):
        """Create a DataLoader from the input data.

        :param x: Input data.
        :param y: Target data.
        :return: DataLoader for data.
        """
        x, y = data
        data = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=self.configuration.batch_size, shuffle=True
        )
        return loader

    def predict(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate outputs and loss.

        :param data: Input data.
        :param labels: Target data.
        :return: Loss.
        """
        outputs = self.model(data)
        ## We only select the last timestep (not generalised for all solutions)
        # This will only predict one label per sequence, not one per timestep.
        loss = self.criterion(outputs, labels[:, -1])
        loss = -torch.log(torch.clamp(loss, min=1e-7))
        return loss

    def validation(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model.

        :param val_loader: DataLoader for validation data.
        :return: Validation Loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                loss = self.predict(data, labels)

                val_loss += loss.item()
        return val_loss

    def train(
        self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader
    ):
        """Train the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        """
        logger.info("Training Started...")
        training_loss = 0
        for epoch in range(self.configuration.num_epochs):
            self.model.train()
            for _i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                loss = self.predict(data, labels)

                # Backward Pass and Optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()
                self.model_io.model_checkpoint()

            val_loss = self.validation(val_loader)
            logger.info(
                f"Epoch: {epoch}, Training Loss: {training_loss}, Validation Loss: {val_loss}"
            )
            if self.configuration.early_stopping and self.early_stopping(training_loss, val_loss):
                logging.info("Early Stopping condition reached")
                break

        logger.info("Training Finished...")

    def run(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ):
        """Run the training Pipeline.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        """
        train_loader = self.make_dataloader(train_data)
        val_loader = self.make_dataloader(val_data)
        self.train(train_loader, val_loader)

        self.model_io.create_model_package(
            filename="./models/anomaly_detection.onnx",
            sequence_length=self.configuration.sequence_length,
            input_size=self.configuration.input_size,
        )
