"""Model listing and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.services import forecaster
from backend.utils.config import get_settings

from backend.routers.predict import verify_api_key

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(
    request: Request,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List trained models and aggregate metrics from metrics_comparison.csv."""
    settings = get_settings()
    models_dir = settings.models_dir
    csv_path = models_dir / "metrics_comparison.csv"
    meta_path = models_dir / "metadata.json"
    if not csv_path.exists():
        raise HTTPException(status_code=503, detail="metrics_comparison.csv not found")
    df = forecaster.list_available_models(csv_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return {
        "metrics": df.to_dict(orient="records"),
        "metadata": meta,
        "models_dir": str(models_dir),
    }


@router.get("/forecast/full")
async def forecast_full(
    request: Request,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return test-set actual vs predicted series for dashboard charts."""
    settings = get_settings()
    models_dir = settings.models_dir
    tf = forecaster.load_test_forecasts(models_dir)
    if tf.empty:
        raise HTTPException(status_code=503, detail="test_forecasts not available")
    return {"rows": tf.to_dict(orient="records")}
