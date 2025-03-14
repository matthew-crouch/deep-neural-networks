"""LSTM model for anomaly detection."""

import logging

import torch
import torch.distributed as dist
import torch.onnx
from pydantic import BaseModel
from torch import nn

from src.anomaly_detection.model_io import ModelIo

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


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
        if abs(val_loss) > abs(train_loss):
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
    early_stopping: bool = False
    rank: int = 0
    world_size: int = 0
    quantise: dict = {"enabled": False, "num_bits": 8}


class TrainingPipeline:
    """Training pipeline for the Anomaly Detection model."""

    def __init__(self, model, configuration: dict):
        """Initialize the training pipeline."""
        self.configuration = TrainingConfig(**configuration)

        self.early_stopping = EarlyStopping()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configuration.learning_rate
        )
        if self.configuration.rank > 1:
            self.setup_distributed_training(
                rank=self.configuration.rank, world_size=self.configuration.world_size
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_size, num_parameters = self.get_model_size_bytes(self.model)
        logger.info(f"Total Model Size in RAM: {round(model_size * 1e-9, 4)} GB")
        logger.info(f"Number of Parameters: {round(num_parameters * 1e-9, 4)} B")

        if self.configuration.quantise.get("enabled"):
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

            model_size, num_parameters = self.get_model_size_bytes(self.model)
            logger.info(
                f"Total Model Size in RAM after Quantisation: {round(model_size * 1e-9, 4)} GB"
            )

        self.model.to(self.device)

        self.model_io = ModelIo(self.model)

    @staticmethod
    def setup_distributed_training(rank: int, world_size: int):
        """Create process group."""
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        """Cleanup the distributed training."""
        dist.destroy_process_group()

    @staticmethod
    def get_model_size_bytes(model: torch.nn.Module) -> tuple:
        """Calculate the model size.

        Calculate total number of bytes occupied by model's params + buffers
        and return the number of parameters in the model.
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()

        buffer_size = 0
        for buf in model.buffers():
            buffer_size += buf.numel() * buf.element_size()

        num_parameters = model.num_parameters
        if type(num_parameters).__name__ == "method":
            num_parameters = model.num_parameters()

        return (param_size + buffer_size, num_parameters)

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
        if len(labels.shape) > 1:
            ## We only select the last timestep (not generalised for all solutions)
            # This will only predict one label per sequence, not one per timestep.
            loss = self.criterion(outputs, labels[:, -1])
        else:
            loss = self.criterion(outputs, labels)
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
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        rank=1,
        world_size=1,
    ):
        """Train the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        """
        logger.info("Training Started...")
        for epoch in range(self.configuration.num_epochs):
            self.model.train()
            for i, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                train_loss = self.predict(data, labels)

                # Backward Pass and Optimization
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                self.model_io.save(filename=f"model_{i}.pth", is_checkpoint=True)

            val_loss = self.validation(val_loader)
            logger.info(
                f"Epoch: {epoch}, Training Loss: {train_loss.item()}, Validation Loss: {val_loss}"
            )
            if self.configuration.early_stopping and self.early_stopping(train_loss, val_loss):
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
        self.model_io.save("anomaly_detection.pth")
