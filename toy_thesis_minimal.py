
from dataclasses import dataclass

import numpy as np
import pandas as pd


ID_COL = "gebernummer"
DATE_COL = "reference_date_int"
ASSET_COL = "F0101_0380_0010"


@dataclass
class ToyConfig:
    n_banks: int = 120
    n_quarters: int = 8
    n_features: int = 100
    n_factors: int = 6
    start_quarter: int = 202203
    expected_missing_per_row: float = 20
    ar_rho: float = 0.65
    random_state: int | None = 0


def simulate_toy_finrep(cfg: ToyConfig = ToyConfig()) -> tuple[pd.DataFrame, list[str]]:
    """
    copy some 
    """

    rng = np.random.default_rng(cfg.random_state)


    # 1. Panel structure: banks and quarters
    year = cfg.start_quarter // 100
    month = cfg.start_quarter % 100
    start = pd.Period(f"{year}-{month:02d}", freq="Q")
    quarters = [
        100 * p.year + 3 * p.quarter
        for p in pd.period_range(start=start, periods=cfg.n_quarters, freq="Q")
    ]

    bank_ids = [f"TOY{i:06d}" for i in range(1, cfg.n_banks + 1)]
    value_cols = [f"F_TOY_{j:04d}" for j in range(1, cfg.n_features + 1)]


    # 2. Total assets A_{i,t}
    bank_size = rng.normal(loc=15.0, scale=1.0, size=cfg.n_banks)
    growth = rng.normal(loc=0.01, scale=0.03, size=(cfg.n_banks, cfg.n_quarters))
    log_assets = bank_size[:, None] + np.cumsum(growth, axis=1)
    total_assets = np.exp(log_assets)

    # 3. Latent factor model for normalized reporting ratios R_{i,t,j}
    baseline = rng.normal(loc=0.06, scale=0.025, size=cfg.n_features)
    loadings = rng.normal(scale=0.08, size=(cfg.n_factors, cfg.n_features))

    scores = np.zeros((cfg.n_banks, cfg.n_quarters, cfg.n_factors))
    scores[:, 0, :] = rng.normal(size=(cfg.n_banks, cfg.n_factors))

    innovation_scale = np.sqrt(1.0 - cfg.ar_rho**2)
    for t in range(1, cfg.n_quarters):
        scores[:, t, :] = (
            cfg.ar_rho * scores[:, t - 1, :]
            + innovation_scale * rng.normal(size=(cfg.n_banks, cfg.n_factors))
        )

    quarter_effects = rng.normal(scale=0.015, size=(cfg.n_quarters, cfg.n_features))
    noise = rng.normal(scale=0.015, size=(cfg.n_banks, cfg.n_quarters, cfg.n_features))

    ratios = baseline + scores @ loadings + quarter_effects[None, :, :] + noise


    # 4. Raw FINREP-like reporting values 
    values = total_assets[:, :, None] * ratios

    if cfg.expected_missing_per_row > 0:
        if cfg.expected_missing_per_row > cfg.n_features:
            raise ValueError(
                "expected_missing_per_row cannot exceed n_features. "
                f"Got {cfg.expected_missing_per_row=} and {cfg.n_features=}."
            )

        # Row-wise missingness: for each bank-quarter report, choose 
        base_missing = int(np.floor(cfg.expected_missing_per_row))
        extra_missing_prob = cfg.expected_missing_per_row - base_missing

        missing_counts = np.full(
            shape=(cfg.n_banks, cfg.n_quarters),
            fill_value=base_missing,
            dtype=int,
        )
        missing_counts += (
            rng.random(size=(cfg.n_banks, cfg.n_quarters)) < extra_missing_prob
        ).astype(int)

        for i in range(cfg.n_banks):
            for t in range(cfg.n_quarters):
                k = missing_counts[i, t]
                if k == 0:
                    continue
                missing_cols = rng.choice(cfg.n_features, size=k, replace=False)
                values[i, t, missing_cols] = np.nan


    # 5. Convert panel arrays to a simple dataframe
    large_bank = total_assets.mean(axis=1) > np.median(total_assets.mean(axis=1)) # not needed anymore

    rows = []
    for i, bank_id in enumerate(bank_ids):
        for t, quarter in enumerate(quarters):
            row = {
                ID_COL: bank_id,
                DATE_COL: quarter,
                ASSET_COL: total_assets[i, t],
                "peer_group": "large" if large_bank[i] else "small",
            }
            row.update({col: values[i, t, j] for j, col in enumerate(value_cols)})
            rows.append(row)

    return pd.DataFrame(rows), value_cols