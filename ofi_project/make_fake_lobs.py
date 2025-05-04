"""
make_fake_lobs.py – Generate multiple synthetic stock LOB CSV files 
(structure identical to LOBSTER format).

Usage:
python make_fake_lobs.py \
       --source first_25000_rows.csv  \
       --n 4                          \
       --price_step 1.5               \
       --price_noise 0.3              \
       --size_noise 0.25              \
       --out_dir data/fakes/
"""

from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import pandas as pd


def make_one_fake(
    df: pd.DataFrame,
    symbol: str,
    price_shift: float,
    price_noise: float,
    size_noise: float,
) -> pd.DataFrame:
    fake = df.copy()

    # 1) Update symbol
    fake["symbol"] = symbol

    # 2) Modify price columns
    price_cols = [c for c in fake.columns if "bid_px" in c or "ask_px" in c]
    for col in price_cols:
        base = fake[col].astype("float64").values
        jitter = np.random.uniform(-price_noise, price_noise, size=len(fake))
        fake[col] = base + price_shift + jitter

    # 3) Modify size columns
    size_cols = [c for c in fake.columns if "bid_sz" in c or "ask_sz" in c]
    for col in size_cols:
        scale = 1 + np.random.uniform(-size_noise, size_noise, size=len(fake))
        fake[col] = (fake[col].astype("float64") * scale).clip(lower=1)

    # 4) Prevent negative spreads (bid ≥ ask)
    depth = max(int(col.split("_")[-1]) for col in price_cols)  # e.g., 10 levels
    for lvl in range(depth):
        bid = f"bid_px_{lvl}"
        ask = f"ask_px_{lvl}"
        if bid in fake.columns and ask in fake.columns:
            wrong = fake[bid] >= fake[ask]
            # Simple fix: raise ask price by 0.01 if necessary
            fake.loc[wrong, ask] = fake.loc[wrong, bid] + 0.01

    return fake


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=3, help="Number of fake stocks to generate")
    p.add_argument(
        "--price_step", type=float, default=1.0, help="Fixed price shift per stock"
    )
    p.add_argument("--price_noise", type=float, default=0.2, help="Random price noise")
    p.add_argument("--size_noise", type=float, default=0.2, help="Random size noise")
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    src = pd.read_csv(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        offset = (i + 1) * args.price_step
        symbol = f"FAKE_{i+1:02d}"
        fake_df = make_one_fake(
            src,
            symbol=symbol,
            price_shift=offset,
            price_noise=args.price_noise,
            size_noise=args.size_noise,
        )

        out_path = out_dir / f"{symbol}.csv"
        fake_df.to_csv(out_path, index=False)
        print(f"[OK] {symbol:>7} -> {out_path}")

    print("All fake LOBs generated.")


if __name__ == "__main__":
    main()
