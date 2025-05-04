from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ofi_integrated(ofi_levels: pd.DataFrame) -> pd.Series:
    """
    Perform PCA integration of multi-level OFI into a single integrated OFI series.

    Parameters
    ----------
    ofi_levels : pd.DataFrame
        DataFrame containing multi-level OFI features (columns = OFI at each depth level).

    Returns
    -------
    tuple[pd.Series, PCA]
        First principal component as a Series (aligned by timestamp),
        and the fitted PCA object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ofi_levels.fillna(0).values)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled)
    return pd.Series(pc1[:, 0], index=ofi_levels.index, name="ofi_integrated"), pca
