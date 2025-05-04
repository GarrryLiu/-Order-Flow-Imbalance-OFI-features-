# OFI Project

A Python library for computing Order Flow Imbalance (OFI) features from Limit Order Book (LOB) data and analyzing cross-impact between stocks.

## Overview

This project implements the methodology described in the paper "Cross-impact of order flow imbalance in equity markets." It processes financial market LOB data to extract OFI features and analyze cross-impact between different stocks. The project supports:

- Best-level OFI calculation
- Multi-level OFI with exponential decay weighting
- Integrated OFI using PCA
- Cross-impact analysis between stocks using LASSO regression

## Requirements

- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- joblib >= 1.3.0

## Usage

### Processing LOB Data

```bash
python -m src.run_pipeline_cross --freq 1min --levels 10 --out_dir result
```

Parameters:
- `--freq`: Resampling frequency (default: 1min)
- `--levels`: Depth levels to use (default: 10)
- `--out_dir`: Output directory (default: result)

### Generating Synthetic Data

```bash
python make_fake_lobs.py --source data/first_25000_rows.csv --n 4
```

Parameters:
- `--source`: Source LOB file
- `--n`: Number of synthetic stocks to generate
- `--price_step`: Fixed price shift per stock
- `--price_noise`: Random price noise
- `--size_noise`: Random size noise
- `--out_dir`: Output directory

## Project Structure

- `src/`: Core library code
  - `loader.py`: LOB data loading utilities
  - `features/`: OFI feature extraction modules
  - `run_pipeline_cross.py`: Main pipeline script
- `data/`: Input LOB data files
- `result/`: Output feature files
