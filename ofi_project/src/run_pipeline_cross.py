from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .loader import load_lob
from .features.ofi_best import ofi_best
from .features.ofi_multi import ofi_multi
from .features.ofi_integrated import ofi_integrated
from .features.ofi_cross import fit_cross_impact


def build_single_stock_features(
    file_path: Path,
    freq: str,
    levels: int,
) -> tuple[str, pd.DataFrame, pd.Series]:
    """
    For one stock LOB file → compute OFI features.
    Returns (symbol, feature DataFrame, ofi_int series).
    """
    lob = load_lob(file_path, levels=levels, freq=freq)  # T × (4k)
    symbol = file_path.stem.upper()

    feats = pd.DataFrame(index=lob.index)

    # (a) Best-level OFI
    feats["ofi_best"] = ofi_best(lob)

    # (b) Multi-level OFI: 10 columns
    multi_df = ofi_multi(lob, k=levels)
    feats = feats.join(multi_df)

    # (c) Integrated OFI via PCA
    ofi_int, _ = ofi_integrated(multi_df)
    feats["ofi_int"] = ofi_int

    return symbol, feats, ofi_int


# -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="1min", help="Resample frequency, e.g., 1min")
    parser.add_argument("--levels", type=int, default=10, help="Depth levels to use")
    parser.add_argument("--out_dir", default="result", help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # STEP 1: Automatically find all CSV files in 'data/'
    # ---------------------------------------------------------------
    data_dir = Path("data")
    file_paths = sorted(data_dir.glob("*.csv"))

    if len(file_paths) < 1:
        raise FileNotFoundError("No CSV files found in the 'data/' folder.")

    print(f"[INFO] Found {len(file_paths)} files:")
    for fp in file_paths:
        print(f"    {fp.name}")

    # ---------------------------------------------------------------
    # STEP 2: Loop through all files and compute features
    # ---------------------------------------------------------------
    all_int_ofi = []  # For cross-impact later
    all_returns = []  # For cross-impact later
    symbols = []

    for fp in file_paths:
        symbol, feat_df, ofi_int = build_single_stock_features(
            fp, args.freq, args.levels
        )
        symbols.append(symbol)

        # Save per-stock feature CSV
        feat_path = out_dir / f"{symbol}_features.csv"
        feat_df.to_csv(feat_path)
        print(f"[OK] {symbol}: {len(feat_df)} rows → {feat_path}")

        # Collect for cross-impact
        all_int_ofi.append(ofi_int.rename(symbol))

        price = load_lob(fp, levels=args.levels, freq=args.freq)["bid_price_1"].ffill()
        ret = np.log(price).diff().fillna(0).rename(symbol)
        all_returns.append(ret)

    # ---------------------------------------------------------------
    # STEP 3: Compute cross-impact β-matrix if more than 1 stock
    # ---------------------------------------------------------------
    if len(symbols) < 2:
        print("[INFO] Only one stock found – skipping cross-impact β matrix.")
        return

    ofi_int_df = pd.concat(all_int_ofi, axis=1)  # T × N
    returns_df = pd.concat(all_returns, axis=1)  # T × N

    betas = fit_cross_impact(returns_df, ofi_int_df)
    beta_path = out_dir / "cross_betas.csv"
    betas.to_csv(beta_path)

    print(f"[OK] β-matrix ({betas.shape[0]} × {betas.shape[1]}) saved to {beta_path}")


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
