import torch
from torch import nn
import torch.nn.functional as F


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


class MistralRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.weight


class MistralRotaryEmbedding(nn.Module):
    """Rotary Embedding Placeholder (For Simplicity)."""

    def forward(self, x):
        return x  # In practice, you'd use actual RoPE embeddings here


class MistralMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) block used in Mistral."""

    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 14336, bias=False)
        self.up_proj = nn.Linear(dim, 14336, bias=False)
        self.down_proj = nn.Linear(14336, dim, bias=False)
        self.act_fn = nn.SiLU()  # Swish activation function

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralAttention(nn.Module):
    """Mistral Attention Block."""

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // 4, bias=False)  # 1024 instead of 4096
        self.v_proj = nn.Linear(dim, dim // 4, bias=False)  # 1024 instead of 4096
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Placeholder forward pass for demonstration purposes
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output = self.o_proj(q + k + v)  # Simplified for illustration
        return attn_output


class MistralDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer in Mistral."""

    def __init__(self, dim):
        super().__init__()
        self.self_attn = MistralAttention(dim)
        self.mlp = MistralMLP(dim)
        self.input_layernorm = MistralRMSNorm(dim)
        self.post_attention_layernorm = MistralRMSNorm(dim)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MistralModel(nn.Module):
    """Mistral Transformer Model."""

    def __init__(self, vocab_size=32000, dim=4096, num_layers=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MistralDecoderLayer(dim) for _ in range(num_layers)])
        self.norm = MistralRMSNorm(dim)
        self.rotary_emb = MistralRotaryEmbedding()  # Rotary embeddings placeholder
        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.rotary_emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
