from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from .utils import timer


def load_lob(
    file: str | Path,
    levels: int = 10,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Load and resample a single-stock limit order book (LOB) file.

    Parameters
    ----------
    file : str or Path
        CSV path to the raw LOB file (e.g., 'first_25000_rows.csv').
    levels : int, default=10
        Depth levels to keep (max 10 for this dataset).
    freq : str, default='1min'
        Resampling frequency (e.g., '1min', '5s').

    Returns
    -------
    DataFrame
        Resampled order book indexed by timestamp.
        For each level i (1 to levels), the columns are:
            - bid_price_i
            - ask_price_i
            - bid_qty_i
            - ask_qty_i
    """

    file = Path(file)
    df = (
        pd.read_csv(file, parse_dates=["ts_event"])
        .rename(columns={"ts_event": "timestamp"})
        .set_index("timestamp")
        .sort_index()
    )

    col_map: dict[str, str] = {}
    for lv in range(levels):
        suf = f"{lv:02d}"  # "00" … "09"
        col_map[f"bid_price_{lv+1}"] = f"bid_px_{suf}"
        col_map[f"ask_price_{lv+1}"] = f"ask_px_{suf}"
        col_map[f"bid_qty_{lv+1}"] = f"bid_sz_{suf}"
        col_map[f"ask_qty_{lv+1}"] = f"ask_sz_{suf}"

    # keep only expected columns (ignore extras)
    df = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
    df = df[~df.index.duplicated(keep="last")]

    with timer("Resampling"):
        price_cols = [c for c in df.columns if "price" in c]
        qty_cols = [c for c in df.columns if "qty" in c]

        # Use the *last* snapshot within each bucket, then forward‑fill
        resampled = pd.concat(
            [
                df[price_cols].resample(freq).last().ffill(),
                df[qty_cols].resample(freq).last().ffill(),
            ],
            axis=1,
        ).astype(np.float32)

    return resampled
