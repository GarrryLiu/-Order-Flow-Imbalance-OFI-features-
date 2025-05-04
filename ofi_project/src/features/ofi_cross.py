from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def fit_cross_impact(
    returns_df: pd.DataFrame,  # (T, N) log-returns
    int_ofi_df: pd.DataFrame,  # (T, N) integrated OFI
    *,
    cv_splits: int = 5,
    alphas: np.ndarray | None = None,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Estimate sparse cross-impact matrix using LASSO regression.

    Model:
    -------
    r_i,t = α_i + Σ_j β_ij · OFI_j,t

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of log-returns for N stocks. Shape (T, N).
    int_ofi_df : pd.DataFrame
        DataFrame of integrated OFI signals for N stocks. Shape (T, N).
    cv_splits : int, optional
        Number of time series cross-validation splits. Default is 5.
    alphas : np.ndarray or None, optional
        Grid of alpha (regularization) values. If None, defaults to logspace(-8, -3, 40).
    scale : bool, optional
        Whether to standardize features and target. Default is True.

    Returns
    -------
    pd.DataFrame
        Cross-impact coefficient matrix (β_ij), with target stocks as rows
        and source OFI signals as columns.
    """

    if alphas is None:
        alphas = np.logspace(-4, 0, 40)

    if scale:
        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(int_ofi_df.fillna(0).values)
        y_scaler = StandardScaler()
    else:
        X = int_ofi_df.fillna(0).values

    betas = pd.DataFrame(
        index=returns_df.columns, columns=int_ofi_df.columns, dtype=float
    )

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for col in returns_df.columns:
        y_raw = returns_df[col].fillna(0).values
        y = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel() if scale else y_raw

        model = LassoCV(
            alphas=alphas,
            cv=tscv,
            fit_intercept=True,
            max_iter=20000,
            n_jobs=-1,
            random_state=42,
        ).fit(X, y)

        if scale:
            beta_raw = model.coef_ * (y_scaler.scale_[0] / x_scaler.scale_)
        else:
            beta_raw = model.coef_

        betas.loc[col] = beta_raw

    return betas
