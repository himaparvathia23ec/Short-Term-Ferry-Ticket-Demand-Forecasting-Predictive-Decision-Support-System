"""Tests for preprocessor and forecaster."""

from pathlib import Path

import numpy as np

from backend.services.preprocessor import (
    fit_transform_scalers,
    preprocess_pipeline,
    time_based_split_indices,
)
from backend.services import forecaster


def test_time_split_ordering():
    n = 100
    tr, te = time_based_split_indices(n, 0.2)
    assert len(tr) + len(te) == n
    assert tr[-1] < te[0]


def test_scaler_fit_on_train_only():
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    tr, te = time_based_split_indices(50, 0.2)
    Xtr, Xte, scaler = fit_transform_scalers(X[tr], X[te])
    assert Xtr.shape == (len(tr), 5)
    assert float(Xtr.min()) >= 0.0 and float(Xtr.max()) <= 1.0


def test_predict_row_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    data_path = root / "data" / "ferry_tickets.csv"
    if not (models_dir / "metadata.json").exists():
        return
    forecaster.clear_feature_cache()
    out = forecaster.predict_row(models_dir, data_path, "2025-09-01T12:00:00", "1hr", "naive")
    assert "sales_forecast" in out
    assert "latency_ms" in out


def test_preprocess_pipeline_runs():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "ferry_tickets.csv"
    if not data_path.exists():
        return
    res, _ = preprocess_pipeline(data_path, test_ratio=0.2)
    assert len(res.df) > 100
