"""Example of How to build a simple LSTM for anomaly detection."""

import torch
from torch import nn


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
        self.linear_1 = nn.Linear(hidden_size, output_size)

        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Pass Function."""
        h_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_init = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # forward pass through LSTM layer, output shape: (batch_size, seq_length, hidden_size)
        output, _ = self.lstm(x.to(self.device), (h_init, c_init))
        output = self.linear_1(output[:, -1, :])

        return output
