"""
Load persisted models and produce forecasts for a given timestamp and horizon.
"""

from __future__ import annotations

import json
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.services.preprocessor import (
    COL_REDEMPTION,
    COL_SALES,
    build_full_feature_table,
    load_scaler,
    preprocess_pipeline,
)


def _snap_to_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Floor timestamp to the previous 15-minute boundary (interval end convention)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.floor("15min")


@lru_cache(maxsize=1)
def _cached_feature_table(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Cache engineered feature matrix in-process (path string for hashability)."""
    full, cols = build_full_feature_table(Path(data_path))
    return full, cols


def clear_feature_cache() -> None:
    """Clear cached feature table (tests)."""
    _cached_feature_table.cache_clear()


def _load_metadata(models_dir: Path) -> Dict[str, Any]:
    with open(models_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _model_filename(model_key: str, target: str, horizon: str, version: str = "v1.0") -> str:
    """Map API model key to on-disk filename."""
    key = model_key.lower().replace("-", "_")
    mapping = {
        "xgboost": "xgboost",
        "random_forest": "random_forest",
        "gradient_boosting": "gradient_boosting",
        "linear_regression": "linear_regression",
        "linear": "linear_regression",
    }
    mk = mapping.get(key, key)
    return f"{mk}_{target}_{horizon}_{version}.pkl"


def load_regressor(models_dir: Path, model_key: str, target: str, horizon: str) -> Any:
    """Load sklearn/xgboost pickle for a target and horizon."""
    fname = _model_filename(model_key, target, horizon)
    path = models_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_row(
    models_dir: Path,
    data_path: Path,
    timestamp: str,
    horizon: str,
    model_key: str,
) -> Dict[str, Any]:
    """
    Predict sales and redemption counts for a horizon using a trained tabular model.

    Returns dict with forecasts, approximate confidence bands, and latency.
    """
    t0 = time.perf_counter()
    meta = _load_metadata(models_dir)
    version = meta.get("version", "v1.0")
    scaler_path = models_dir / f"feature_scaler_{version}.pkl"
    if not scaler_path.exists():
        scaler_path = models_dir / "feature_scaler_v1.0.pkl"

    ts = _snap_to_15min(pd.Timestamp(timestamp))
    full, feature_cols = _cached_feature_table(str(data_path.resolve()))
    if ts not in full.index:
        prior = full.index[full.index <= ts]
        if len(prior) == 0:
            raise ValueError("Timestamp is before dataset range.")
        ts = prior[-1]

    mk = model_key.lower().replace("-", "_")

    if mk == "naive":
        sales_hat = float(full.loc[ts, COL_SALES])
        red_hat = float(full.loc[ts, COL_REDEMPTION])
    elif mk in ("moving_average_4", "ma4", "ma_4"):
        sales_hat = float(full.loc[ts, f"{COL_SALES}_roll_mean_4"])
        red_hat = float(full.loc[ts, f"{COL_REDEMPTION}_roll_mean_4"])
    else:
        scaler = load_scaler(scaler_path)
        row = full.loc[ts, feature_cols]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if row.isna().any():
            raise ValueError("Insufficient history to build features for this timestamp.")
        X = row.values.astype(np.float32).reshape(1, -1)
        Xs = scaler.transform(X)
        sales_model = load_regressor(models_dir, model_key, "sales", horizon)
        red_model = load_regressor(models_dir, model_key, "redemption", horizon)
        sales_hat = float(sales_model.predict(Xs)[0])
        red_hat = float(red_model.predict(Xs)[0])

    residual = meta.get("residual_quantiles", {})
    res_m = residual.get(mk, {})
    s_key = f"sales_{horizon}"
    r_key = f"redemption_{horizon}"
    s_off = res_m.get(s_key, {}) if isinstance(res_m, dict) else {}
    r_off = res_m.get(r_key, {}) if isinstance(res_m, dict) else {}
    slo, shi = s_off.get("low_offset", 0.0), s_off.get("high_offset", 0.0)
    rlo, rhi = r_off.get("low_offset", 0.0), r_off.get("high_offset", 0.0)

    sales_low = max(0.0, sales_hat + slo)
    sales_high = max(0.0, sales_hat + shi)
    red_low = max(0.0, red_hat + rlo)
    red_high = max(0.0, red_hat + rhi)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "sales_forecast": round(sales_hat, 2),
        "redemption_forecast": round(red_hat, 2),
        "confidence_lower": [round(sales_low, 2), round(red_low, 2)],
        "confidence_upper": [round(sales_high, 2), round(red_high, 2)],
        "horizon": horizon,
        "model": model_key,
        "latency_ms": round(latency_ms, 2),
        "aligned_timestamp": str(ts),
    }


def list_available_models(metrics_csv: Path) -> pd.DataFrame:
    """Read metrics comparison CSV."""
    return pd.read_csv(metrics_csv)


def load_test_forecasts(models_dir: Path) -> pd.DataFrame:
    """Load parquet of test-set predictions."""
    p = models_dir / "test_forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def build_test_export_dataframe(
    models_dir: Path,
    data_path: Path,
    model_key: str,
    horizon: str,
    n_intervals: int,
) -> pd.DataFrame:
    """
    Build actual vs predicted rows for the first ``n_intervals`` test timestamps.

    Recomputes predictions for the selected model and horizon (not the static
    ``test_forecasts.parquet``, which is XGBoost 1hr only).
    """
    result, _ = preprocess_pipeline(data_path, test_ratio=0.2)
    df = result.df
    train_idx = result.train_idx
    test_idx = result.test_idx
    feature_columns = result.feature_columns

    meta = _load_metadata(models_dir)
    version = meta.get("version", "v1.0")
    scaler_path = models_dir / f"feature_scaler_{version}.pkl"
    if not scaler_path.exists():
        scaler_path = models_dir / "feature_scaler_v1.0.pkl"

    n = min(max(1, n_intervals), len(test_idx))
    rows = test_idx[:n]

    X = df[feature_columns].values.astype(np.float32)
    scaler = load_scaler(scaler_path)
    X_test = scaler.transform(X[test_idx])
    X_slice = X_test[:n]

    ys = f"y_sales_{horizon}"
    yr = f"y_redemption_{horizon}"
    actual_s = df[ys].iloc[rows].values
    actual_r = df[yr].iloc[rows].values
    ts = df.index[rows]

    mk = model_key.lower().replace("-", "_")

    if mk == "naive":
        pred_s = df[COL_SALES].iloc[rows].values.astype(float)
        pred_r = df[COL_REDEMPTION].iloc[rows].values.astype(float)
    elif mk in ("moving_average_4", "ma4", "ma_4"):
        pred_s = df[f"{COL_SALES}_roll_mean_4"].iloc[rows].values.astype(float)
        pred_r = df[f"{COL_REDEMPTION}_roll_mean_4"].iloc[rows].values.astype(float)
    else:
        m_s = load_regressor(models_dir, model_key, "sales", horizon)
        m_r = load_regressor(models_dir, model_key, "redemption", horizon)
        pred_s = m_s.predict(X_slice)
        pred_r = m_r.predict(X_slice)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            f"actual_sales_{horizon}": actual_s,
            f"pred_sales_{horizon}": pred_s,
            f"actual_redemption_{horizon}": actual_r,
            f"pred_redemption_{horizon}": pred_r,
            "model": model_key,
            "horizon": horizon,
        }
    )
    return out
