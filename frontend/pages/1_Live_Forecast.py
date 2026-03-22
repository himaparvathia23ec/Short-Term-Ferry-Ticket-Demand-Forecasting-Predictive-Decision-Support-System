"""Live forecast dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import get_setting

ROOT = Path(__file__).resolve().parents[2]


def api_base() -> str:
    b = get_setting("API_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000"
    return b.rstrip("/")


@st.cache_data(show_spinner=False)
def load_local_series() -> pd.DataFrame:
    p = ROOT / "data" / "ferry_tickets.csv"
    df = pd.read_csv(p)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").set_index("Timestamp")
    return df


def align_timestamp_to_index(full: pd.DataFrame, ts: pd.Timestamp) -> pd.Timestamp:
    """Use the latest row at or before ``ts`` so charts match the forecast reference."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = pd.Timestamp(ts.replace(tzinfo=None))
    if ts in full.index:
        return ts
    prior = full.index[full.index <= ts]
    if len(prior) == 0:
        return full.index[0]
    return prior[-1]


def history_window(full: pd.DataFrame, ts: pd.Timestamp, periods: int = 48) -> pd.DataFrame:
    """Last ``periods`` intervals ending at the aligned reference time."""
    end = align_timestamp_to_index(full, ts)
    return full.loc[:end].tail(periods)


def rolling_stats_at(full: pd.DataFrame, ts: pd.Timestamp, col: str, window: int = 96) -> tuple[float, float]:
    """Mean and std of rolling ``window`` at the aligned reference time (for surge heuristic)."""
    end = align_timestamp_to_index(full, ts)
    sub = full.loc[:end, col]
    roll = sub.rolling(window, min_periods=min(8, window))
    m = float(roll.mean().iloc[-1]) if len(sub) else 0.0
    s = float(roll.std().iloc[-1]) or 1.0
    return m, s


st.title("Toronto Island Ferry — Demand Intelligence")
st.caption("Short-horizon demand forecasting for operations")

with st.sidebar:
    ts_raw = st.text_input("Reference time (ISO, interval end)", value="2025-09-01T14:00:00")
    ts = pd.Timestamp(ts_raw)
    horizon = st.radio("Horizon", ["15min", "30min", "1hr", "2hr"], horizontal=True)
    model = st.selectbox(
        "Model",
        ["xgboost", "random_forest", "gradient_boosting", "linear_regression", "naive", "moving_average_4"],
    )
    target_v = st.radio("Display", ["Both", "Sales", "Redemptions"], horizontal=True)

try:
    r = httpx.post(
        f"{api_base()}/predict",
        json={"timestamp": pd.Timestamp(ts).isoformat(), "horizon": horizon, "model": model},
        timeout=60.0,
    )
    r.raise_for_status()
    out = r.json()
except Exception:
    st.warning("API unavailable — showing local heuristic preview only.")
    out = {
        "sales_forecast": 0.0,
        "redemption_forecast": 0.0,
        "confidence_lower": [0.0, 0.0],
        "confidence_upper": [0.0, 0.0],
        "horizon": horizon,
    }

sales_f = float(out.get("sales_forecast", 0))
red_f = float(out.get("redemption_forecast", 0))

full = load_local_series()
last_mean, std = rolling_stats_at(full, ts, "Sales Count", window=96)
z = (sales_f - last_mean) / std if std else 0.0
level = "Critical" if z > 2 else "High" if z > 1 else "Medium" if z > 0 else "Low"
acc = max(0.0, min(100.0, 100.0 - min(abs(z) * 5, 40)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecasted Sales", f"{sales_f:.1f}")
c2.metric("Forecasted Redemptions", f"{red_f:.1f}")
c3.metric("Heuristic accuracy %", f"{acc:.1f}")
c4.metric("Demand level", level)

if z > 2:
    st.error("Demand surge alert — forecast is more than 2σ above the rolling mean at the reference time.")

hist = history_window(full, ts, 48)
t_end = align_timestamp_to_index(full, ts)
if hist.empty:
    st.warning("No history on or before the reference time — pick a later timestamp.")
else:
    fig = go.Figure()
    if target_v in ("Both", "Sales"):
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist["Sales Count"], name="Actual sales", line=dict(color="#00b4d8"))
        )
    if target_v in ("Both", "Redemptions"):
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist["Redemption Count"],
                name="Actual redemptions",
                line=dict(color="#7ee787"),
            )
        )
    fig.add_vline(x=t_end, line_dash="dash", line_color="#888")
    fig.update_layout(
        template="plotly_dark",
        title=f"48 intervals ending at reference ({t_end})",
        height=480,
        xaxis_title="Time (interval end)",
        yaxis_title="Count",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "History window matches the forecast: last row is the latest interval ≤ reference time. "
        "Adjust the sidebar time to slide this window."
    )

lo = out.get("confidence_lower", [0, 0])
hi = out.get("confidence_upper", [0, 0])
st.caption(f"Confidence bands (residual quantiles): sales [{lo[0]:.1f}, {hi[0]:.1f}], redemptions [{lo[1]:.1f}, {hi[1]:.1f}]")
