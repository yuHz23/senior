"""Alternative autoencoder architectures for ablation study.

Models:
  LSTMAutoencoder  — same structure as GRUAutoencoder but with LSTM cells
  MLPAutoencoder   — simple feedforward baseline (no temporal modelling)
"""
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder — identical structure to GRUAutoencoder."""

    def __init__(self, input_size, hidden_size=128, bottleneck_size=32,
                 seq_len=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len

        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_size)

        self.decoder_fc   = nn.Linear(bottleneck_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        return torch.tanh(self.encoder_fc(h_n[-1]))

    def decode(self, z):
        h0 = torch.tanh(self.decoder_fc(z))
        repeated = h0.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(repeated)
        return self.output_fc(out)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))


class MLPAutoencoder(nn.Module):
    """Feedforward MLP Autoencoder — no temporal modelling (baseline).

    Flattens the window to a 1-D vector, encodes to bottleneck, reconstructs.
    """

    def __init__(self, input_size, hidden_size=128, bottleneck_size=32,
                 seq_len=64, **kwargs):
        super().__init__()
        self.seq_len    = seq_len
        self.input_size = input_size
        flat = seq_len * input_size

        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, flat),
        )

    def encode(self, x):
        return self.encoder(x.reshape(x.size(0), -1))

    def decode(self, z):
        return self.decoder(z).reshape(-1, self.seq_len, self.input_size)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))
