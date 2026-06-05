
import numpy as np
import pandas as pd

ID_COL = "gebernummer"
DATE_COL = "reference_date_int"
ASSET_COL = "F0101_0380_0010"


def value_columns(df, asset_col=ASSET_COL):
    skip = {ID_COL, DATE_COL, asset_col, "peer_group"}
    return [c for c in df.columns if str(c).startswith("F") and c not in skip]


def previous_quarter(q):
    year, month = divmod(int(q), 100)
    return (year - 1) * 100 + 12 if month == 3 else year * 100 + month - 3


def ratios(df, cols, asset_col=ASSET_COL):
    A = pd.to_numeric(df[asset_col], errors="coerce")
    R = df[cols].apply(pd.to_numeric, errors="coerce").div(A, axis=0)
    return R.replace([np.inf, -np.inf], np.nan), A


def inject_anomalies(
    clean_data,
    *,
    target_quarter,
    cols=None,
    peer_row_fraction=0.03,
    temporal_row_fraction=0.03,
    feature_fraction=0.15,
    random_state=0,
    id_col=ID_COL,
    date_col=DATE_COL,
    asset_col=ASSET_COL,
    group_col="peer_group",
):
    """
lacks currently fh
    """
    rng = np.random.default_rng(random_state)

    prev_q = previous_quarter(target_quarter)
    target = clean_data.loc[clean_data[date_col] == target_quarter].copy().reset_index(drop=True)
    previous = clean_data.loc[clean_data[date_col] == prev_q].copy().reset_index(drop=True)

    cols = list(cols) if cols is not None else value_columns(target, asset_col)
    R_t, A_t = ratios(target, cols, asset_col)
    R_p, _A_p = ratios(previous, cols, asset_col)

    labels = target[[id_col, date_col]].copy()
    labels["anomaly_type"] = "none"
    labels["is_peer_anomaly"] = False
    labels["is_temporal_anomaly"] = False
    labels["is_any_anomaly"] = False
    labels["n_changed_cells"] = 0

    cells = pd.DataFrame("none", index=target.index, columns=cols)
    prev_row = {bank: i for i, bank in previous[id_col].items()}

    n_peer = round(peer_row_fraction * len(target))
    n_temporal = round(temporal_row_fraction * len(target))
    n_cols = max(1, round(feature_fraction * len(cols)))

    used = set()

    peer_rows = _choose(rng, target.index, n_peer)
    for i in peer_rows:
        changed = []
        same_group = _same_group(target, i, group_col)

        for c in _choose(rng, cols, n_cols):
            donors = [j for j in same_group if j != i and pd.notna(R_t.at[j, c])]
            if not donors:
                continue
            j = rng.choice(donors)
            target.at[i, c] = A_t.at[i] * R_t.at[j, c]
            changed.append(c)

        _mark(labels, cells, i, changed, "peer")
        used.add(i)

    temporal_candidates = [i for i in target.index if i not in used and target.at[i, id_col] in prev_row]
    temporal_rows = _choose(rng, temporal_candidates, n_temporal)
    for i in temporal_rows:
        p_i = prev_row[target.at[i, id_col]]
        changed = []
        same_group = _same_group(target, i, group_col)

        for c in _choose(rng, cols, n_cols):
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

        _mark(labels, cells, i, changed, "temporal")

    summary = pd.DataFrame(
        {
            "quantity": [
                "target_quarter",
                "previous_quarter",
                "target_rows",
                "peer_rows",
                "temporal_rows",
                "changed_cells",
            ],
            "value": [
                int(target_quarter),
                int(prev_q),
                len(target),
                int(labels["is_peer_anomaly"].sum()),
                int(labels["is_temporal_anomaly"].sum()),
                int((cells != "none").sum().sum()),
            ],
        }
    )

    return {
        "data": target,
        "previous_data": previous,
        "row_labels": labels,
        "cell_labels": cells,
        "summary": summary,
    }


def _same_group(df, row, group_col):
    if group_col in df.columns:
        return df.index[df[group_col] == df.at[row, group_col]]
    return df.index


def _choose(rng, items, n):
    items = list(items)
    if n <= 0 or not items:
        return []
    return list(rng.choice(items, size=min(int(n), len(items)), replace=False))


def _mark(labels, cells, row, cols, kind):
    if not cols:
        return
    labels.at[row, "anomaly_type"] = kind
    labels.at[row, f"is_{kind}_anomaly"] = True
    labels.at[row, "is_any_anomaly"] = True
    labels.at[row, "n_changed_cells"] = len(cols)
    cells.loc[row, cols] = kind
