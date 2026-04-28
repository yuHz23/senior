import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.lstm_autoencoder import GRUAutoencoder


def train(
    model: GRUAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 5,
    device: str = "cpu",
) -> dict:
    """Train the GRU Autoencoder on Normal-only windows.

    Args:
        X_train: (N, window_size, n_features)
        X_val:   (M, window_size, n_features)
        save_path: path to save the best model checkpoint

    Returns:
        dict with train_losses and val_losses lists.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_losses.append(epoch_loss / len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += criterion(model(batch), batch).item() * len(batch)
        val_losses.append(val_loss / len(val_ds))

        print(f"Epoch {epoch:3d} | train={train_losses[-1]:.6f} | val={val_losses[-1]:.6f}")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Best val loss: {best_val_loss:.6f} — saved to {save_path}")
    return {"train_losses": train_losses, "val_losses": val_losses}
