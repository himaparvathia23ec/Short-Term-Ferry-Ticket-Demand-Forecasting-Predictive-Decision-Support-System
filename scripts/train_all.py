#!/usr/bin/env python3
"""
Train all baseline, ML, and time-series models; persist artifacts under models/.

Run from project root: python scripts/train_all.py
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project root
ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from backend.services.preprocessor import (  # noqa: E402
    COL_REDEMPTION,
    COL_SALES,
    HORIZON_STEPS,
    fit_transform_scalers,
    load_raw_ferry_csv,
    preprocess_pipeline,
    reindex_to_15min_grid,
    save_json,
    save_scaler,
    time_based_split_indices,
    winsorize_iqr_by_hour,
)
from backend.utils.config import reset_settings_cache  # noqa: E402

HORIZONS = ["15min", "30min", "1hr", "2hr"]
TARGET_PREFIX = {"sales": "y_sales_", "redemption": "y_redemption_"}


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE with epsilon denominator to avoid divide-by-zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def add_metrics_rows(
    rows: List[Dict[str, Any]],
    target: str,
    horizon: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mape_score(y_true, y_pred)
    rows.append(
        {
            "target": target,
            "horizon": horizon,
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
    )


def train_tabular_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    factory: Callable[[], Any],
    models_dir: Path,
    target: str,
    horizon: str,
    version: str,
) -> Tuple[np.ndarray, Any]:
    """Fit sklearn/xgb model and save pickle; return test predictions and estimator."""
    model = factory()
    if model_name == "xgboost":
        n = len(X_train)
        split = max(int(n * 0.9), n - 5000)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fname = f"{model_name}_{target}_{horizon}_{version}.pkl"
    path = models_dir / fname
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return pred, model


def run_sarima_stride(
    series: pd.Series,
    test_idx: np.ndarray,
    horizon_steps: int,
    max_refits: int = 100,
    tail: int = 3500,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward SARIMA on strided test indices; returns (y_true, y_pred) aligned."""
    if os.environ.get("SKIP_SARIMA", "").lower() in ("1", "true", "yes"):
        return None
    try:
        from pmdarima import auto_arima
    except (ImportError, ValueError):
        return None

    step = max(len(test_idx) // max_refits, 96)
    positions = test_idx[::step][:max_refits]
    ys: List[float] = []
    ps: List[float] = []
    for pos in positions:
        if pos + horizon_steps >= len(series):
            continue
        hist = series.iloc[: pos + 1].iloc[-tail:]
        if len(hist) < horizon_steps + 96:
            continue
        try:
            m = auto_arima(
                hist.values,
                seasonal=True,
                m=96,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=2,
                max_q=2,
                max_P=1,
                max_Q=1,
                max_d=1,
                max_D=1,
                max_order=8,
                information_criterion="aic",
            )
            fc = m.predict(n_periods=horizon_steps)
            pred = float(fc[-1])
        except Exception:
            continue
        actual = float(series.iloc[pos + horizon_steps])
        ys.append(actual)
        ps.append(pred)
    if len(ys) < 5:
        return None
    return np.array(ys), np.array(ps)


def run_prophet_stride(
    series: pd.Series,
    test_idx: np.ndarray,
    horizon_steps: int,
    freq: str = "15min",
    max_refits: int = 80,
    tail: int = 4000,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    from prophet import Prophet

    step = max(len(test_idx) // max_refits, 96)
    positions = test_idx[::step][:max_refits]
    ys: List[float] = []
    ps: List[float] = []
    for pos in positions:
        if pos + horizon_steps >= len(series):
            continue
        hist = series.iloc[: pos + 1].iloc[-tail:]
        if len(hist) < horizon_steps + 48:
            continue
        # Prophet expects timezone-naive timestamps
        ix = pd.DatetimeIndex(hist.index)
        ds = ix.astype("datetime64[ns]")
        df_p = pd.DataFrame({"ds": ds, "y": hist.values})
        try:
            m = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=horizon_steps, freq=freq, include_history=False)
            if horizon_steps == 1:
                fut = m.predict(future.iloc[[0]])
            else:
                fut = m.predict(future)
            pred = float(fut["yhat"].iloc[-1])
        except Exception:
            continue
        actual = float(series.iloc[pos + horizon_steps])
        ys.append(actual)
        ps.append(pred)
    if len(ys) < 5:
        return None
    return np.array(ys), np.array(ps)


def optuna_tune_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 12,
) -> xgb.XGBRegressor:
    import optuna

    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": 42,
        }
        scores = []
        for tr, va in tscv.split(X_train):
            model = xgb.XGBRegressor(**params, early_stopping_rounds=30)
            model.fit(
                X_train[tr],
                y_train[tr],
                eval_set=[(X_train[va], y_train[va])],
                verbose=False,
            )
            pred = model.predict(X_train[va])
            scores.append(mean_absolute_error(y_train[va], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["random_state"] = 42
    return xgb.XGBRegressor(**best, early_stopping_rounds=40)


def export_plotly_dual_timeseries(raw: pd.DataFrame, reports_dir: Path) -> None:
    df = raw.reset_index()
    df.columns = ["timestamp", COL_SALES, COL_REDEMPTION]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df[COL_SALES], name="Sales", line=dict(color="#00b4d8")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df[COL_REDEMPTION], name="Redemption", line=dict(color="#7ee787")),
        row=2,
        col=1,
    )
    fig.update_layout(
        template="plotly_dark",
        title="Toronto Island Ferry — Demand Intelligence — Sales & Redemption",
        height=720,
    )
    path_html = reports_dir / "sales_redemption_timeseries.html"
    fig.write_html(str(path_html))
    try:
        fig.write_image(str(reports_dir / "sales_redemption_timeseries.png"), width=1400, height=720)
    except Exception:
        pass


def export_hourly_heatmap(df: pd.DataFrame, col: str, title: str, reports_dir: Path, fname: str) -> None:
    tmp = df.copy()
    tmp["hour"] = tmp.index.hour
    tmp["dow"] = tmp.index.day_name()
    g = tmp.groupby(["dow", "hour"], observed=False)[col].mean().reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g["dow"] = pd.Categorical(g["dow"], categories=order, ordered=True)
    pivot = g.pivot(index="dow", columns="hour", values=col)
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Weekday", color="Avg count"),
        title=title,
        color_continuous_scale="Teal",
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark")
    fig.write_html(str(reports_dir / f"{fname}.html"))
    try:
        fig.write_image(str(reports_dir / f"{fname}.png"), width=1000, height=500)
    except Exception:
        pass


def main() -> None:
    reset_settings_cache()
    data_path = ROOT / "data" / "ferry_tickets.csv"
    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    result, _ = preprocess_pipeline(data_path, test_ratio=0.2)
    df = result.df
    train_idx, test_idx = result.train_idx, result.test_idx
    feature_columns = result.feature_columns

    cap = os.environ.get("FERRY_TRAIN_MAX_ROWS")
    if cap:
        nmax = int(cap)
        df = df.iloc[-nmax:].copy()
        train_idx, test_idx = time_based_split_indices(len(df), 0.2)
        result.train_idx, result.test_idx = train_idx, test_idx
        result.train_end_time = df.index[train_idx[-1]]

    # Also build clean series for Prophet/SARIMA from same pipeline
    raw = load_raw_ferry_csv(data_path)
    gridded, _ = reindex_to_15min_grid(raw)
    clean_series = winsorize_iqr_by_hour(gridded)

    export_plotly_dual_timeseries(clean_series, reports_dir)
    export_hourly_heatmap(
        clean_series,
        COL_SALES,
        "Toronto Island Ferry — Demand Intelligence — Avg sales by hour × weekday",
        reports_dir,
        "hourly_heatmap",
    )

    X = df[feature_columns].values.astype(np.float32)
    y_blocks: Dict[Tuple[str, str], np.ndarray] = {}
    for h in HORIZONS:
        y_blocks[("sales", h)] = df[f"y_sales_{h}"].values.astype(np.float32)
        y_blocks[("redemption", h)] = df[f"y_redemption_{h}"].values.astype(np.float32)

    X_train, X_test = X[train_idx], X[test_idx]
    scaler_path = models_dir / "feature_scaler_v1.0.pkl"
    X_train_s, X_test_s, scaler = fit_transform_scalers(X_train, X_test)
    save_scaler(scaler, scaler_path)

    rows: List[Dict[str, Any]] = []
    residual_store: Dict[str, Any] = {}
    version = "v1.0"

    for horizon in HORIZONS:
        h_steps = HORIZON_STEPS[horizon]
        for target in ("sales", "redemption"):
            y = y_blocks[(target, horizon)]
            y_train, y_test = y[train_idx], y[test_idx]
            key_base = f"{target}_{horizon}"

            # Naive: current level of the series predicts multi-step persistence
            level_col = COL_SALES if target == "sales" else COL_REDEMPTION
            naive_pred = df[level_col].iloc[test_idx].values.astype(float)
            add_metrics_rows(rows, target, horizon, "naive", y_test, naive_pred)
            residual_store.setdefault("naive", {})[key_base] = _residual_quantiles(y_test, naive_pred)

            ma_col = f"{level_col}_roll_mean_4"
            ma_pred = df[ma_col].iloc[test_idx].values.astype(float)
            add_metrics_rows(rows, target, horizon, "moving_average_4", y_test, ma_pred)
            residual_store.setdefault("moving_average_4", {})[key_base] = _residual_quantiles(y_test, ma_pred)

            pred_lr, _ = train_tabular_models(
                X_train_s,
                X_test_s,
                y_train,
                y_test,
                "linear_regression",
                LinearRegression,
                models_dir,
                target,
                horizon,
                version,
            )
            add_metrics_rows(rows, target, horizon, "linear_regression", y_test, pred_lr)
            residual_store.setdefault("linear_regression", {})[key_base] = _residual_quantiles(y_test, pred_lr)

            pred_rf, _ = train_tabular_models(
                X_train_s,
                X_test_s,
                y_train,
                y_test,
                "random_forest",
                lambda: RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    n_jobs=-1,
                    random_state=42,
                ),
                models_dir,
                target,
                horizon,
                version,
            )
            add_metrics_rows(rows, target, horizon, "random_forest", y_test, pred_rf)
            residual_store.setdefault("random_forest", {})[key_base] = _residual_quantiles(y_test, pred_rf)

            pred_gbr, _ = train_tabular_models(
                X_train_s,
                X_test_s,
                y_train,
                y_test,
                "gradient_boosting",
                lambda: GradientBoostingRegressor(random_state=42),
                models_dir,
                target,
                horizon,
                version,
            )
            add_metrics_rows(rows, target, horizon, "gradient_boosting", y_test, pred_gbr)
            residual_store.setdefault("gradient_boosting", {})[key_base] = _residual_quantiles(y_test, pred_gbr)

            pred_xgb, _ = train_tabular_models(
                X_train_s,
                X_test_s,
                y_train,
                y_test,
                "xgboost",
                lambda: xgb.XGBRegressor(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    early_stopping_rounds=40,
                ),
                models_dir,
                target,
                horizon,
                version,
            )
            add_metrics_rows(rows, target, horizon, "xgboost", y_test, pred_xgb)
            residual_store.setdefault("xgboost", {})[key_base] = _residual_quantiles(y_test, pred_xgb)

            # SARIMA / Prophet on strided evaluation
            s_series = clean_series[COL_SALES] if target == "sales" else clean_series[COL_REDEMPTION]
            s_series = s_series.reindex(df.index).ffill()
            aligned = s_series.values
            series_pd = pd.Series(aligned, index=df.index)

            sar = run_sarima_stride(series_pd, test_idx, h_steps)
            if sar is not None:
                yt, yp = sar
                add_metrics_rows(rows, target, horizon, "sarima", yt, yp)
                residual_store.setdefault("sarima", {})[key_base] = _residual_quantiles(yt, yp)
            else:
                rows.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "model": "sarima",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                    }
                )

            pr = run_prophet_stride(series_pd, test_idx, h_steps)
            if pr is not None:
                yt, yp = pr
                add_metrics_rows(rows, target, horizon, "prophet", yt, yp)
                residual_store.setdefault("prophet", {})[key_base] = _residual_quantiles(yt, yp)
            else:
                rows.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "model": "prophet",
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(models_dir / "metrics_comparison.csv", index=False)

    # Optuna tune xgboost on 1hr sales as representative
    y_tr_s = y_blocks[("sales", "1hr")][train_idx]
    tuned = optuna_tune_xgb(X_train_s, y_tr_s, n_trials=10)
    n = len(X_train_s)
    split = max(int(n * 0.9), n - 5000)
    tuned.fit(
        X_train_s[:split],
        y_tr_s[:split],
        eval_set=[(X_train_s[split:], y_tr_s[split:])],
        verbose=False,
    )
    with open(models_dir / f"xgboost_tuned_sales_1hr_{version}.pkl", "wb") as f:
        pickle.dump(tuned, f)

    # SHAP for xgboost sales 1hr
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(tuned)
        sample = X_test_s[: min(800, len(X_test_s))]
        shap_vals = explainer.shap_values(sample)
        shap.summary_plot(
            shap_vals,
            sample,
            feature_names=feature_columns,
            show=False,
            plot_type="bar",
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary_sales.png", dpi=150, bbox_inches="tight")
        plt.close()

        y_tr_r = y_blocks[("redemption", "1hr")][train_idx]
        tuned_r = optuna_tune_xgb(X_train_s, y_tr_r, n_trials=10)
        tuned_r.fit(
            X_train_s[:split],
            y_tr_r[:split],
            eval_set=[(X_train_s[split:], y_tr_r[split:])],
            verbose=False,
        )
        explainer_r = shap.TreeExplainer(tuned_r)
        shap_vals_r = explainer_r.shap_values(sample)
        shap.summary_plot(
            shap_vals_r,
            sample,
            feature_names=feature_columns,
            show=False,
            plot_type="bar",
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary_redemption.png", dpi=150, bbox_inches="tight")
        plt.close()
        with open(models_dir / f"xgboost_tuned_redemption_1hr_{version}.pkl", "wb") as f:
            pickle.dump(tuned_r, f)
    except Exception as e:
        print("SHAP skipped:", e)

    # Best model per horizon/target + forecast vs actual charts (xgboost default)
    best_table = []
    for horizon in HORIZONS:
        for target in ("sales", "redemption"):
            sub = metrics_df[(metrics_df.horizon == horizon) & (metrics_df.target == target)]
            sub = sub.dropna(subset=["mae"])
            if sub.empty:
                continue
            best_row = sub.iloc[int(sub["mae"].argmin())]
            best_table.append(best_row.to_dict())
            model_name = str(best_row["model"])
            y_col = f"y_{target}_{horizon}"
            y_test = df[y_col].iloc[test_idx].values
            if model_name in (
                "linear_regression",
                "random_forest",
                "gradient_boosting",
                "xgboost",
            ):
                fname = f"{model_name}_{target}_{horizon}_{version}.pkl"
                with open(models_dir / fname, "rb") as f:
                    mdl = pickle.load(f)
                pred = mdl.predict(X_test_s)
            elif model_name == "naive":
                level_col = COL_SALES if target == "sales" else COL_REDEMPTION
                pred = df[level_col].iloc[test_idx].values.astype(float)
            elif model_name == "moving_average_4":
                level_col = COL_SALES if target == "sales" else COL_REDEMPTION
                pred = df[f"{level_col}_roll_mean_4"].iloc[test_idx].values.astype(float)
            else:
                pred = np.full_like(y_test, np.nan, dtype=float)

            ts = df.index[test_idx]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=y_test, name="Actual", line=dict(color="#7ee787")))
            fig.add_trace(go.Scatter(x=ts, y=pred, name="Forecast", line=dict(color="#00b4d8")))
            fig.update_layout(
                template="plotly_dark",
                title=f"Toronto Island Ferry — Forecast vs Actual ({model_name} / {target} / {horizon})",
            )
            safe = model_name.replace(" ", "_")
            fig.write_html(str(reports_dir / f"forecast_vs_actual_{safe}_{target}_{horizon}.html"))
            try:
                fig.write_image(
                    str(reports_dir / f"forecast_vs_actual_{safe}_{target}_{horizon}.png"),
                    width=1200,
                    height=500,
                )
            except Exception:
                pass

    # Metrics comparison bar chart
    fig_m = px.bar(
        metrics_df.dropna(subset=["mae"]),
        x="model",
        y="mae",
        color="horizon",
        barmode="group",
        facet_col="target",
        title="Toronto Island Ferry — Demand Intelligence — MAE by model",
    )
    fig_m.update_layout(template="plotly_dark")
    fig_m.write_html(str(reports_dir / "model_metrics_comparison.html"))
    try:
        fig_m.write_image(str(reports_dir / "model_metrics_comparison.png"), width=1400, height=600)
    except Exception:
        pass

    # Confidence bands example (xgboost sales 1hr)
    with open(models_dir / f"xgboost_sales_1hr_{version}.pkl", "rb") as f:
        xgb_s = pickle.load(f)
    pred_full = xgb_s.predict(X_test_s)
    y_full = df["y_sales_1hr"].iloc[test_idx].values
    bounds = residual_store["xgboost"]["sales_1hr"]
    low = pred_full + bounds["low_offset"]
    high = pred_full + bounds["high_offset"]
    ts = df.index[test_idx]
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=ts, y=y_full, name="Actual", line=dict(color="#fff")))
    fig_c.add_trace(go.Scatter(x=ts, y=pred_full, name="Pred", line=dict(color="#00b4d8")))
    fig_c.add_trace(
        go.Scatter(
            x=list(ts) + list(ts)[::-1],
            y=list(high) + list(low)[::-1],
            fill="toself",
            fillcolor="rgba(0,180,216,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Band",
        )
    )
    fig_c.update_layout(
        template="plotly_dark",
        title="Toronto Island Ferry — Demand Intelligence — Confidence bands (XGBoost sales 1hr)",
    )
    fig_c.write_html(str(reports_dir / "confidence_bands_chart.html"))
    try:
        fig_c.write_image(str(reports_dir / "confidence_bands_chart.png"), width=1200, height=550)
    except Exception:
        pass

    # Test forecasts parquet for API
    with open(models_dir / f"xgboost_sales_1hr_{version}.pkl", "rb") as f:
        xgb_1h = pickle.load(f)
    ps = xgb_1h.predict(X_test_s)
    tf = pd.DataFrame(
        {
            "timestamp": df.index[test_idx],
            "actual_sales_1hr": df["y_sales_1hr"].iloc[test_idx].values,
            "pred_sales_1hr": ps,
            "actual_redemption_1hr": df["y_redemption_1hr"].iloc[test_idx].values,
        }
    )
    with open(models_dir / f"xgboost_redemption_1hr_{version}.pkl", "rb") as f:
        xgb_r = pickle.load(f)
    tf["pred_redemption_1hr"] = xgb_r.predict(X_test_s)
    tf.to_parquet(models_dir / "test_forecasts.parquet", index=False)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path.relative_to(ROOT)),
        "feature_columns": feature_columns,
        "train_end_time": str(result.train_end_time),
        "scaler_path": str(scaler_path.relative_to(ROOT)),
        "metrics_csv": "metrics_comparison.csv",
        "residual_quantiles": residual_store,
        "best_per_target_horizon": best_table,
        "horizons": HORIZONS,
        "model_file_pattern": "{model}_{target}_{horizon}_v1.0.pkl",
    }
    save_json(metadata, models_dir / "metadata.json")
    with open(models_dir / "residual_bounds.json", "w", encoding="utf-8") as f:
        json.dump(residual_store, f, indent=2, default=str)

    print("Training complete. Metrics:", models_dir / "metadata.json")


def _residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    res = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "low_offset": float(np.percentile(res, 10)),
        "high_offset": float(np.percentile(res, 90)),
    }


if __name__ == "__main__":
    main()
