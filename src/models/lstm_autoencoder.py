import torch
import torch.nn as nn


class GRUAutoencoder(nn.Module):
    """GRU-based Autoencoder for semantic compression and anomaly detection.

    Encoder compresses a (batch, seq_len, input_size) sequence into a
    bottleneck vector of shape (batch, bottleneck_size). Decoder reconstructs
    the original sequence from the bottleneck vector.

    Only the bottleneck vector needs to be transmitted (semantic communication).
    Reconstruction error (MSE) at the decoder is used for anomaly detection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        seq_len: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size

        self.encoder_gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_size)

        self.decoder_fc = nn.Linear(bottleneck_size, hidden_size)
        self.decoder_gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → z: (batch, bottleneck_size)"""
        _, h_n = self.encoder_gru(x)          # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]                       # (batch, hidden)
        return torch.tanh(self.encoder_fc(h_last))  # tanh keeps z in [-1, 1]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, bottleneck_size) → x_hat: (batch, seq_len, input_size)"""
        h0 = torch.tanh(self.decoder_fc(z))   # (batch, hidden)
        # Repeat the context vector across time steps
        repeated = h0.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden)
        out, _ = self.decoder_gru(repeated)
        return self.output_fc(out)             # (batch, seq_len, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction x_hat with same shape as x."""
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error. Returns shape (batch,)."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))
