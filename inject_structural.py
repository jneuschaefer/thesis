from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "gebernummer"
DATE_COL = "reference_date_int"
ASSET_COL = "F0101_0380_0010"


def value_columns(df, asset_col=ASSET_COL):
    """FINREP value columns used for ratios X / total assets."""
    skip = {ID_COL, DATE_COL, asset_col, "peer_group"}
    return [c for c in df.columns if str(c).startswith("F") and c not in skip]


def previous_quarter(q):
    year, month = divmod(int(q), 100)
    return (year - 1) * 100 + 12 if month == 3 else year * 100 + month - 3


def ratios(df, cols, asset_col=ASSET_COL):
    A = pd.to_numeric(df[asset_col], errors="coerce")
    R = df[cols].apply(pd.to_numeric, errors="coerce").div(A, axis=0)
    R.loc[A <= 0, :] = np.nan
    return R.replace([np.inf, -np.inf], np.nan), A


def _prepare(clean_data, target_quarter, cols, id_col, date_col, asset_col):
    prev_q = previous_quarter(target_quarter)
    target = clean_data.loc[clean_data[date_col] == target_quarter].copy().reset_index(drop=True)
    previous = clean_data.loc[clean_data[date_col] == prev_q].copy().reset_index(drop=True)

    if target.empty:
        raise ValueError(f"No rows for target_quarter={target_quarter}.")
    if previous.empty:
        raise ValueError(f"No rows for previous quarter {prev_q}.")

    cols = list(cols) if cols is not None else value_columns(target, asset_col)
    R_t, A_t = ratios(target, cols, asset_col)
    R_p, A_p = ratios(previous, cols, asset_col)

    labels = target[[id_col, date_col]].copy()
    labels["anomaly_type"] = "none"
    labels["is_peer_anomaly"] = False
    labels["is_temporal_anomaly"] = False
    labels["is_any_anomaly"] = False
    labels["n_changed_cells"] = 0

    cells = pd.DataFrame("none", index=target.index, columns=cols)
    return target, previous, cols, R_t, A_t, R_p, A_p, labels, cells


def _pick(rng, items, n):
    items = list(items)
    if n <= 0 or not items:
        return []
    return list(rng.choice(items, size=min(n, len(items)), replace=False))


def _mark(labels, cells, row, cols, kind):
    labels.at[row, "anomaly_type"] = kind
    labels.at[row, f"is_{kind}_anomaly"] = True
    labels.at[row, "is_any_anomaly"] = True
    labels.at[row, "n_changed_cells"] = len(cols)
    cells.loc[row, cols] = kind


def _finish(target, previous, labels, cells, target_quarter, kind):
    return {
        "data": target,
        "previous_data": previous,
        "row_labels": labels,
        "cell_labels": cells,
        "summary": pd.DataFrame(
            {
                "quantity": [
                    "target_quarter",
                    "previous_quarter",
                    "target_rows",
                    f"{kind}_rows",
                    "changed_cells",
                ],
                "value": [
                    int(target_quarter),
                    int(previous_quarter(target_quarter)),
                    len(target),
                    int(labels[f"is_{kind}_anomaly"].sum()),
                    int((cells != "none").sum().sum()),
                ],
            }
        ),
    }


def inject_peer_anomaly(
    clean_data,
    *,
    target_quarter,
    cols=None,
    row_fraction=0.03,
    feature_fraction=0.15,
    random_state=0,
    id_col=ID_COL,
    date_col=DATE_COL,
    asset_col=ASSET_COL,
    group_col="peer_group",
):
    """
    Structural peer anomaly.

    For selected target-quarter banks, replace a subset of current ratios by
    ratios copied from other banks in the same quarter.

    Each copied cell is realistic marginally, but the row no longer belongs to
    one coherent latent-factor bank profile.
    """
    rng = np.random.default_rng(random_state)
    target, previous, cols, R_t, A_t, *_rest = _prepare(
        clean_data, target_quarter, cols, id_col, date_col, asset_col
    )
    labels, cells = _rest[-2], _rest[-1]

    n_rows = round(row_fraction * len(target))
    n_cols = max(1, round(feature_fraction * len(cols)))

    for i in _pick(rng, target.index, n_rows):
        if pd.isna(A_t.at[i]) or A_t.at[i] <= 0:
            continue

        same_group = target.index
        if group_col in target:
            same_group = target.index[target[group_col] == target.at[i, group_col]]

        changed = []
        for c in _pick(rng, cols, n_cols):
            donors = [j for j in same_group if j != i and pd.notna(R_t.at[j, c])]
            if not donors:
                continue
            j = rng.choice(donors)
            target.at[i, c] = A_t.at[i] * R_t.at[j, c]
            changed.append(c)

        if changed:
            _mark(labels, cells, i, changed, "peer")

    return _finish(target, previous, labels, cells, target_quarter, "peer")


def inject_temporal_anomaly(
    clean_data,
    *,
    target_quarter,
    cols=None,
    row_fraction=0.03,
    feature_fraction=0.15,
    random_state=0,
    id_col=ID_COL,
    date_col=DATE_COL,
    asset_col=ASSET_COL,
    group_col="peer_group",
):
    """
    Structural temporal anomaly.

    For selected banks, replace a subset of their quarterly ratio changes by
    changes copied from other banks:

        R_new(i,t,c) = R(i,t-1,c) + [R(j,t,c) - R(j,t-1,c)]

    Each copied change is realistic marginally, but the bank's own temporal
    evolution no longer follows one coherent latent-factor path.
    """
    rng = np.random.default_rng(random_state)
    target, previous, cols, R_t, A_t, R_p, _A_p, labels, cells = _prepare(
        clean_data, target_quarter, cols, id_col, date_col, asset_col
    )

    prev_row = {bank: i for i, bank in previous[id_col].items()}
    valid_rows = [i for i in target.index if target.at[i, id_col] in prev_row]

    n_rows = round(row_fraction * len(valid_rows))
    n_cols = max(1, round(feature_fraction * len(cols)))

    for i in _pick(rng, valid_rows, n_rows):
        p_i = prev_row[target.at[i, id_col]]
        if pd.isna(A_t.at[i]) or A_t.at[i] <= 0:
            continue

        same_group = target.index
        if group_col in target:
            same_group = target.index[target[group_col] == target.at[i, group_col]]

        changed = []
        for c in _pick(rng, cols, n_cols):
            donors = [
                j for j in same_group
                if j != i
                and target.at[j, id_col] in prev_row
                and pd.notna(R_t.at[j, c])
                and pd.notna(R_p.at[prev_row[target.at[j, id_col]], c])
                and pd.notna(R_p.at[p_i, c])
            ]
            if not donors:
                continue

            j = rng.choice(donors)
            p_j = prev_row[target.at[j, id_col]]
            donor_change = R_t.at[j, c] - R_p.at[p_j, c]
            target.at[i, c] = A_t.at[i] * (R_p.at[p_i, c] + donor_change)
            changed.append(c)

        if changed:
            _mark(labels, cells, i, changed, "temporal")

    return _finish(target, previous, labels, cells, target_quarter, "temporal")
