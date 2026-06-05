"""Optional PyTorch autoencoder detector.

intentionally compact. It is meant as the common-interface wrapper into
which a more specialized BaFin AE implementation can later be
plugged without changing the rest of the thesis pipeline.
"""

import numpy as np
import pandas as pd

from .base import OutlierDetector


class AutoencoderDetector(OutlierDetector):
    """
    Dense autoencoder reconstruction-error detector.

    Requires torch.

    Higher score = more anomalous.
    """

    method_name = "Autoencoder"

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 0,
        verbose: bool = False,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame) -> "AutoencoderDetector":
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

        class AE(nn.Module):
            def __init__(self, n_features: int, hidden_dim: int, latent_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_features),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        self.model_ = AE(n_features, self.hidden_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss(reduction="mean")

        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        self.training_loss_ = []
        for epoch in range(self.epochs):
            losses = []
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
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
            recon = self.model_(x).numpy()
        residual = X_arr - recon
        return np.sqrt((residual ** 2).sum(axis=1))

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("not fitted.")
