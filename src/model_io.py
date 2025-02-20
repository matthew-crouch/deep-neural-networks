"""Model I/O Module."""

import os
import uuid
from pathlib import Path

import torch


class ModelIo:
    """Model I/O class."""

    def __init__(self, model: torch.nn.Module):
        """Initialize the model I/O object."""
        self.model = model
        self.path = Path(f"model_checkpoints/{str(uuid.uuid4())}")
        Path.mkdir(self.path, parents=True, exist_ok=True)

    def load(self, filename: str) -> torch.nn.Module:
        """Load the model from a file."""
        model = torch.load(filename)
        return model

    def save(self, filename: str):
        """Save the model to a file."""
        torch.save(self.model, filename)

    def model_checkpoint(self):
        """Save the model checkpoint."""
        torch.save(self.model.state_dict(), f"{self.path}/model_{uuid.uuid4()}.pth")

    def create_model_package(self, filename: str, sequence_length: int, input_size: int) -> None:
        """Package the model after training to onnx.

        :param filename: Name of the file to save the model.
        """
        self.model.eval()

        if not os.path.exists("models"):
            os.makedirs("models")

        torch.onnx.export(
            self.model,
            torch.randn(1, sequence_length, input_size),
            filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
            },
            opset_version=11,
        )
