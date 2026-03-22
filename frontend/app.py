"""
Toronto Island Ferry — Demand Intelligence (Streamlit entry).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from settings import get_setting

ROOT = Path(__file__).resolve().parent.parent


def inject_css() -> None:
    """Apply dark navy / teal operations theme."""
    st.markdown(
        """
<style>
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 0 12px rgba(0,180,216,0.08);
    }
    div[data-testid="stMetric"]:hover { box-shadow: 0 0 16px rgba(0,180,216,0.18); transition: 0.2s; }
    h1, h2, h3 { color: #f0f6fc !important; font-weight: 600; }
    .block-container { padding-top: 1.2rem; }
</style>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Toronto Island Ferry — Demand Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

app_password = get_setting("APP_PASSWORD")

if app_password and not st.session_state.authenticated:
    st.title("Toronto Island Ferry — Demand Intelligence")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Unlock"):
        if pwd == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()

st.session_state.authenticated = True

pages = [
    st.Page(str(ROOT / "frontend/pages/1_Live_Forecast.py"), title="Live Forecast", icon="🏠"),
    st.Page(str(ROOT / "frontend/pages/2_Model_Comparison.py"), title="Model Comparison", icon="📊"),
    st.Page(str(ROOT / "frontend/pages/3_Historical_Insights.py"), title="Historical Insights", icon="📈"),
    st.Page(str(ROOT / "frontend/pages/4_Export_Reports.py"), title="Export & Reports", icon="📤"),
]

st.navigation(pages).run()
