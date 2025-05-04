import pandas as pd
import numpy as np


def ofi_best(lob: pd.DataFrame) -> pd.Series:
    """
    Compute best-level Order Flow Imbalance (OFI) from the LOB data.

    Parameters
    ----------
    lob : pd.DataFrame
        Limit order book dataframe with bid_qty_1 and ask_qty_1 columns.

    Returns
    -------
    pd.Series
        Series containing best-level OFI indexed by timestamps.
    """
    bid = lob["bid_qty_1"]
    ask = lob["ask_qty_1"]
    ofi = bid.diff().fillna(0) - ask.diff().fillna(0)
    return ofi.astype(np.float32).rename("ofi_best")
