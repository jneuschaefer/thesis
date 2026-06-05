"""Method-overlap """

import pandas as pd


def top_k_overlap(
    rankings: dict[str, pd.DataFrame],
    *,
    k: int,
    id_column: str = "gebernummer",
    date_column: str = "reference_date_int",
) -> pd.DataFrame:
    """Pairwise top-k overlap between detectors."""
    methods = list(rankings.keys())
    top_sets = {}

    for method, df in rankings.items():
        top = df.sort_values("rank").head(k)
        top_sets[method] = set(zip(top[id_column], top[date_column].astype(str)))

    rows = []
    for m1 in methods:
        for m2 in methods:
            intersection = len(top_sets[m1] & top_sets[m2])
            union = len(top_sets[m1] | top_sets[m2])
            rows.append(
                {
                    "method_1": m1,
                    "method_2": m2,
                    "top_k": int(k),
                    "overlap_count": int(intersection),
                    "jaccard": intersection / union if union else 0.0,
                }
            )

    return pd.DataFrame(rows)
