"""
FastAPI application entrypoint for ferry demand forecasting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import models as models_router
from backend.routers import predict as predict_router
from backend.utils.config import get_settings
from backend.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(title="Ferry Demand Forecast API", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Liveness and model artifact check."""
    models_dir = settings.models_dir
    meta = models_dir / "metadata.json"
    return {"status": "ok", "model_loaded": meta.exists()}


app.include_router(predict_router.router, prefix="")
app.include_router(models_router.router, prefix="")
