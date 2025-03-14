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

class AutoEncoder(nn.Module):
    def __init__(self, input_size: torch.tensor, dropout: float):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3072),
            nn.ReLU(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3072, 1024),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 3072),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),
            nn.ReLU(),
            nn.Linear(3072, input_size),
        )

        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x