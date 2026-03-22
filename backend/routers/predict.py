"""Prediction and export routes."""

from __future__ import annotations

import io
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from backend.services import forecaster
from backend.utils.config import get_settings

router = APIRouter(tags=["predict"])


class PredictBody(BaseModel):
    """Request body for /predict."""

    timestamp: str = Field(..., description="ISO timestamp ending a 15-minute interval")
    horizon: Literal["15min", "30min", "1hr", "2hr"] = "1hr"
    model: str = Field("xgboost", description="Model key, e.g. xgboost, random_forest, naive")


class ExportBody(BaseModel):
    """Request body for /export."""

    horizon: Literal["15min", "30min", "1hr", "2hr"] = "1hr"
    model: str = "xgboost"
    n_intervals: int = Field(24, ge=1, le=48, description="Number of future 15-min steps to export")


def verify_api_key(request: Request) -> None:
    """Optional dependency: validate X-API-Key when API_KEY is configured."""
    settings = get_settings()
    expected = settings.api_key
    if not expected:
        return
    got = request.headers.get("X-API-Key")
    if got != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/predict")
async def predict(
    body: PredictBody,
    request: Request,
    _: None = Depends(verify_api_key),
) -> dict:
    """Return sales/redemption forecasts with residual-based confidence bands."""
    settings = get_settings()
    models_dir = settings.models_dir
    data_path = settings.data_path
    try:
        out = forecaster.predict_row(
            models_dir,
            data_path,
            body.timestamp,
            body.horizon,
            body.model,
        )
        logger.info(
            "predict ts={} horizon={} model={} sales={} redemption={} ms={}",
            out.get("aligned_timestamp"),
            body.horizon,
            body.model,
            out.get("sales_forecast"),
            out.get("redemption_forecast"),
            out.get("latency_ms"),
        )
        return out
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@router.post("/export")
async def export_csv(
    body: ExportBody,
    request: Request,
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream a CSV of test-set actual vs predicted rows for the chosen model and horizon."""
    settings = get_settings()
    models_dir = settings.models_dir
    data_path = settings.data_path
    try:
        chunk = forecaster.build_test_export_dataframe(
            models_dir,
            data_path,
            body.model,
            body.horizon,
            body.n_intervals,
        )
        buf = io.StringIO()
        chunk.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ferry_forecast_export.csv"'},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("export failed")
        raise HTTPException(status_code=500, detail="Export failed") from e
