"""
Shared feature engineering and scaling for ferry demand forecasting.

Strict temporal ordering: no shuffling. Train scalers only on the training slice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Canonical column names after rename
COL_SALES = "sales"
COL_REDEMPTION = "redemption"
COL_TIME = "timestamp"

HORIZON_STEPS: Dict[str, int] = {
    "15min": 1,
    "30min": 2,
    "1hr": 4,
    "2hr": 8,
}

RAW_SALES = "Sales Count"
RAW_REDEMPTION = "Redemption Count"
RAW_TS = "Timestamp"


@dataclass
class PreprocessResult:
    """Container for engineered dataframe and train/test index positions."""

    df: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray
    feature_columns: List[str]
    target_columns: List[str]
    train_end_time: pd.Timestamp


def load_raw_ferry_csv(path: Path | str) -> pd.DataFrame:
    """
    Load ferry tickets CSV and normalize column names.

    Expects columns: _id, Timestamp, Redemption Count, Sales Count (order flexible).
    """
    p = Path(path)
    df = pd.read_csv(p)
    rename = {}
    if RAW_TS in df.columns:
        rename[RAW_TS] = COL_TIME
    if RAW_SALES in df.columns:
        rename[RAW_SALES] = COL_SALES
    if RAW_REDEMPTION in df.columns:
        rename[RAW_REDEMPTION] = COL_REDEMPTION
    df = df.rename(columns=rename)
    if COL_TIME not in df.columns:
        raise ValueError(f"Missing timestamp column in {p}")
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], utc=False)
    df = df.sort_values(COL_TIME).drop_duplicates(subset=[COL_TIME], keep="last")
    df = df.set_index(COL_TIME)
    return df[[COL_SALES, COL_REDEMPTION]]


def reindex_to_15min_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Reindex to a complete 15-minute frequency and linearly interpolate inner gaps.

    Returns the reindexed frame and count of inserted NaN slots before interpolation.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    before = len(df)
    reindexed = df.reindex(full_idx)
    missing = int(reindexed[COL_SALES].isna().sum())
    reindexed = reindexed.interpolate(method="linear", limit_direction="both")
    reindexed.index.name = COL_TIME
    return reindexed, missing


def winsorize_iqr_by_hour(
    df: pd.DataFrame,
    columns: Tuple[str, ...] = (COL_SALES, COL_REDEMPTION),
    k: float = 1.5,
) -> pd.DataFrame:
    """
    Cap outliers per time-of-day bucket (0–23) using IQR on each column.

    Preserves temporal continuity by winsorizing rather than dropping rows.
    """
    out = df.copy()
    hour = out.index.hour
    for col in columns:
        capped = out[col].astype(float).copy()
        for h in range(24):
            mask = hour == h
            if not mask.any():
                continue
            vals = capped.loc[mask]
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            capped.loc[mask] = vals.clip(lower=lo, upper=hi)
        out[col] = capped
    return out


def add_temporal_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, DOW, month, weekend, Canadian federal holiday flag."""
    ca_hols = holidays.country_holidays("CA")
    ts = idx
    is_holiday = pd.Series([d in ca_hols for d in ts.date], index=ts)
    return pd.DataFrame(
        {
            "hour_of_day": ts.hour,
            "day_of_week": ts.dayofweek,
            "month": ts.month,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "is_holiday": is_holiday.astype(int),
        },
        index=ts,
    )


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features t-1,t-2,t-4,t-8 and rolling stats (4 and 8 periods) for sales/redemption.
    """
    out = df.copy()
    for col in (COL_SALES, COL_REDEMPTION):
        for lag in (1, 2, 4, 8):
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for w in (4, 8):
            r = out[col].rolling(window=w, min_periods=w)
            out[f"{col}_roll_mean_{w}"] = r.mean()
            out[f"{col}_roll_std_{w}"] = r.std()
            out[f"{col}_roll_max_{w}"] = r.max()
    # Demand surge: z-score of total demand (sales + redemption) vs 8-period rolling mean/std
    total = out[COL_SALES] + out[COL_REDEMPTION]
    rm = total.rolling(8, min_periods=8).mean()
    rs = total.rolling(8, min_periods=8).std().replace(0, np.nan)
    z = (total - rm) / rs
    out["demand_surge_flag"] = (z > 2).fillna(0).astype(int)
    denom = out[COL_REDEMPTION].replace(0, np.nan)
    out["sales_to_redemption_ratio"] = (out[COL_SALES] / denom).replace([np.inf, -np.inf], np.nan)
    out["sales_to_redemption_ratio"] = out["sales_to_redemption_ratio"].fillna(0.0)
    return out


def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Targets: future sales/redemption at +1,+2,+4,+8 steps."""
    out = df.copy()
    shifts = {"15min": 1, "30min": 2, "1hr": 4, "2hr": 8}
    for name, s in shifts.items():
        out[f"y_sales_{name}"] = out[COL_SALES].shift(-s)
        out[f"y_redemption_{name}"] = out[COL_REDEMPTION].shift(-s)
    return out


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Combine engineered columns; returns frame with features + targets and column lists.
    """
    temporal = add_temporal_features(df.index)
    merged = df.join(temporal, how="left")
    merged = add_lag_and_rolling_features(merged)
    merged = add_multi_horizon_targets(merged)

    exclude = {
        COL_SALES,
        COL_REDEMPTION,
        "y_sales_15min",
        "y_sales_30min",
        "y_sales_1hr",
        "y_sales_2hr",
        "y_redemption_15min",
        "y_redemption_30min",
        "y_redemption_1hr",
        "y_redemption_2hr",
    }
    feature_columns = [c for c in merged.columns if c not in exclude]
    target_columns = [c for c in merged.columns if c.startswith("y_")]
    return merged, feature_columns, target_columns


def time_based_split_indices(n: int, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Return train and test positional indices (ordered)."""
    split = int(n * (1 - test_ratio))
    split = max(split, 1)
    split = min(split, n - 1)
    all_idx = np.arange(n)
    return all_idx[:split], all_idx[split:]


def fit_transform_scalers(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler on train only; transform both."""
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    return X_tr, X_te, scaler


def build_full_feature_table(path: Path | str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load data and return the full engineered feature matrix (before dropping NaN rows).

    Used at inference time to look up a feature row aligned to a 15-minute timestamp.
    """
    raw = load_raw_ferry_csv(path)
    gridded, _ = reindex_to_15min_grid(raw)
    clean = winsorize_iqr_by_hour(gridded)
    full, feature_cols, _ = build_feature_matrix(clean)
    return full, feature_cols


def preprocess_pipeline(
    path: Path | str,
    test_ratio: float = 0.2,
) -> Tuple[PreprocessResult, pd.DataFrame]:
    """
    Full pipeline: load, grid, winsorize, features, split indices (no scaling).

    Scaling is applied separately in training/inference using persisted scalers.
    """
    raw = load_raw_ferry_csv(path)
    gridded, _ = reindex_to_15min_grid(raw)
    clean = winsorize_iqr_by_hour(gridded)
    full, feature_cols, target_cols = build_feature_matrix(clean)
    full = full.dropna()
    n = len(full)
    train_idx, test_idx = time_based_split_indices(n, test_ratio)
    train_end_time = full.index[train_idx[-1]]
    result = PreprocessResult(
        df=full,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_columns=feature_cols,
        target_columns=target_cols,
        train_end_time=train_end_time,
    )
    return result, gridded


def horizon_to_target_names(horizon: str) -> Tuple[str, str]:
    """Map API horizon key to y column names for sales and redemption."""
    if horizon not in HORIZON_STEPS:
        raise ValueError(f"Invalid horizon: {horizon}")
    return f"y_sales_{horizon}", f"y_redemption_{horizon}"


def save_scaler(scaler: MinMaxScaler, path: Path) -> None:
    """Persist sklearn scaler."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> MinMaxScaler:
    """Load sklearn scaler from disk."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
