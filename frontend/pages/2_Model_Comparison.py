"""Model comparison view."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import get_setting

ROOT = Path(__file__).resolve().parents[2]


def api_base() -> str:
    b = get_setting("API_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000"
    return b.rstrip("/")


st.title("Model comparison")
metrics_path = ROOT / "models" / "metrics_comparison.csv"
if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    st.dataframe(df, use_container_width=True)
    dplot = df.dropna(subset=["mae"])
    # Separate charts avoid cramped facet titles / overlapping category labels
    c1, c2 = st.columns(2)
    for col, tgt, title in (
        (c1, "sales", "Sales — MAE by model"),
        (c2, "redemption", "Redemptions — MAE by model"),
    ):
        sub = dplot[dplot["target"] == tgt]
        if sub.empty:
            continue
        fig = px.bar(
            sub,
            x="model",
            y="mae",
            color="horizon",
            barmode="group",
            title=title,
        )
        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(b=120),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45),
        )
        fig.update_xaxes(tickangle=-35)
        col.plotly_chart(fig, use_container_width=True)
    best = (
        df.dropna(subset=["mae"])
        .sort_values("mae")
        .groupby(["target", "horizon"], as_index=False)
        .head(1)
    )
    st.subheader("Best model badges (lowest MAE)")
    st.dataframe(best, use_container_width=True)
else:
    st.info("Train models to populate metrics_comparison.csv")

try:
    r = httpx.get(f"{api_base()}/models", timeout=30.0)
    if r.status_code == 200:
        with st.expander("API metadata (`GET /models`)", expanded=False):
            st.json(r.json())
except Exception:
    pass

shap_sales = ROOT / "reports" / "figures" / "shap_summary_sales.png"
if shap_sales.exists():
    st.subheader("SHAP — sales (1hr, tuned XGBoost)")
    st.caption("SHAP plots use a light background from matplotlib; values are unchanged.")
    st.image(str(shap_sales), use_container_width=True)
shap_r = ROOT / "reports" / "figures" / "shap_summary_redemption.png"
if shap_r.exists():
    st.subheader("SHAP — redemptions")
    st.image(str(shap_r), use_container_width=True)
