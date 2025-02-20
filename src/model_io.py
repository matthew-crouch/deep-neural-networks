"""Model I/O Module."""

import os
import uuid
from pathlib import Path

import onnx
import torch


class ModelIo:
    """Model I/O class."""

    def __init__(self, model: torch.nn.Module = None):
        """Initialize the model I/O object."""
        self.model = model
        self.run_id = uuid.uuid4().hex

    def load(self, filename: str) -> torch.nn.Module:
        """Load the model from a file."""
        model = onnx.load(filename)
        return model

    def save(self, filename: str, is_checkpoint: bool = False) -> None:
        """Save the model to a file."""
        if is_checkpoint:
            _model = self.model.state_dict()
            path = Path(f"model_checkpoints/{self.run_id}")
        else:
            _model = self.model
            path = Path("models")

        path.mkdir(parents=True, exist_ok=True)
        torch.save(_model, Path(f"{path}/{filename}"))

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
