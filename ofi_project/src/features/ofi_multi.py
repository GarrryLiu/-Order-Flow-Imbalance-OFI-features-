from __future__ import annotations
import pandas as pd
import numpy as np


def ofi_multi(
    lob: pd.DataFrame,
    k: int = 10,
    decay: float = 5.0,
) -> pd.DataFrame:
    """
    Compute multi-level OFI for k depth levels, applying exponential decay weighting.

    Parameters
    ----------
    lob : pd.DataFrame
        Limit order book dataframe with bid_qty_i and ask_qty_i for i=1 to k.
    k : int, optional
        Number of depth levels to compute OFI for (default is 10).
    decay : float, optional
        Decay factor controlling how fast the weight decreases with level depth.

    Returns
    -------
    pd.DataFrame
        DataFrame with k columns ('ofi_multi_1', ..., 'ofi_multi_k') containing
        weighted OFI values for each depth level.
    """
    weights = np.exp(-np.arange(1, k + 1) / decay)

    bids = lob[[f"bid_qty_{i}" for i in range(1, k + 1)]].diff().fillna(0).values
    asks = lob[[f"ask_qty_{i}" for i in range(1, k + 1)]].diff().fillna(0).values
    diff = bids - asks  # shape (T, k)

    data = {}
    for i in range(k):
        data[f"ofi_multi_{i+1}"] = diff[:, i] * weights[i]

    return pd.DataFrame(data, index=lob.index)
