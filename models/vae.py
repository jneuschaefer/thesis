"""PyTorch variational autoencoder detector."""
#already cleaned version

import numpy as np
import pandas as pd

from .base import OutlierDetector


class VAEDetector(OutlierDetector):
    """
    Variational autoencoder detector.

    Score currently combines reconstruction error and KL contribution:

        score = reconstruction_error + beta * KL

    torch
    """

    method_name = "VAE"

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        beta: float = 1.0,
        batch_size: int = 64,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame) -> "VAEDetector":
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("error") from exc

        torch.manual_seed(self.random_state)
        self.torch_ = torch
        self.nn_ = nn
        self.columns_ = list(X.columns)
        X_arr = X.astype("float32").to_numpy()
        n_features = X_arr.shape[1]

        class VAE(nn.Module):
            def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(hidden_dim, latent_dim)
                self.logvar = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_features),
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.mu(h), self.logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                return recon, mu, logvar

        self.model_ = VAE(n_features, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        self.training_loss_ = []
        for epoch in range(self.epochs):
            losses = []
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, mu, logvar = self.model_(batch)
                recon_loss = ((recon - batch) ** 2).mean()
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else np.nan
            self.training_loss_.append(mean_loss)
            if self.verbose and (epoch + 1) % 25 == 0:
                print(f"epoch={epoch+1}, loss={mean_loss:.6f}")

        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        torch = self.torch_
        X_arr = X.loc[:, self.columns_].astype("float32").to_numpy()

        self.model_.eval()
        with torch.no_grad():
            x = torch.tensor(X_arr)
            mu, logvar = self.model_.encode(x)
            recon = self.model_.decode(mu)
            recon_np = recon.numpy()
            kl_np = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).numpy()

        residual = X_arr - recon_np
        recon_score = np.sqrt((residual ** 2).sum(axis=1))
        return recon_score + self.beta * kl_np

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("not fitted")
