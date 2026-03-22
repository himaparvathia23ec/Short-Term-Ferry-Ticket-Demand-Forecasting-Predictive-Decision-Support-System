"""Export and reports."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import get_setting

ROOT = Path(__file__).resolve().parents[2]


def api_base() -> str:
    b = get_setting("API_URL", "http://127.0.0.1:8000") or "http://127.0.0.1:8000"
    return b.rstrip("/")


@st.cache_data(show_spinner="Building export (first run may take a minute on full data)…")
def _local_export_df(model: str, horizon: str, n: int) -> pd.DataFrame:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.services.forecaster import build_test_export_dataframe
    from backend.utils.config import get_settings

    s = get_settings()
    return build_test_export_dataframe(s.models_dir, s.data_path, model, horizon, n)


st.title("Export & reports")

model = st.selectbox(
    "Model",
    [
        "xgboost",
        "random_forest",
        "gradient_boosting",
        "linear_regression",
        "naive",
        "moving_average_4",
    ],
)
horizon = st.selectbox("Horizon", ["15min", "30min", "1hr", "2hr"])
n = st.slider("Intervals (1–48)", 1, 48, 12)

df_out: pd.DataFrame | None = None
csv_text: str | None = None

try:
    r = httpx.post(
        f"{api_base()}/export",
        json={"model": model, "horizon": horizon, "n_intervals": n},
        timeout=120.0,
    )
    if r.status_code == 200:
        csv_text = r.text
        df_out = pd.read_csv(io.StringIO(r.text))
    else:
        st.warning(r.text)
except Exception:
    csv_text = None
    df_out = None

if df_out is None:
    try:
        df_out = _local_export_df(model, horizon, n)
        csv_text = df_out.to_csv(index=False)
        st.caption("Using local export (API unreachable or error) — same logic as `/export`.")
    except Exception as e:
        st.error(f"Export failed: {e}")

if df_out is not None and csv_text is not None:
    st.dataframe(df_out, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=csv_text,
        file_name=f"ferry_forecast_{model}_{horizon}.csv",
        mime="text/csv",
        key=f"dl_{model}_{horizon}_{n}",
    )
    ps = f"pred_sales_{horizon}"
    if ps in df_out.columns:
        st.metric(f"Mean predicted sales ({horizon})", f"{df_out[ps].mean():.2f}")
