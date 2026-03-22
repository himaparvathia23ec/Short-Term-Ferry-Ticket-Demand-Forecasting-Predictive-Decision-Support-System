```
╔══════════════════════════════════════════════════════════════════╗
║  TORONTO ISLAND FERRY — DEMAND INTELLIGENCE                      ║
║  Short-term ticket forecasting & decision support                 ║
╚══════════════════════════════════════════════════════════════════╝
```

## Problem statement

Ferry operations to Toronto Island Park need **short-horizon** forecasts (15 minutes to 2 hours) for ticket **sales** and **redemptions** so staff can schedule vessels, plan crowd flow, and respond to demand surges safely.

## Live demo

- **Streamlit (placeholder):** `https://share.streamlit.io/<your-org>/ferry-demand-forecast`  
  Replace with your Streamlit Community Cloud URL after deployment.

## Features

- Multi-horizon regression (15m / 30m / 1h / 2h) for sales and redemptions  
- Baselines, tree models, boosting, Prophet; optional SARIMA via `pmdarima`  
- FastAPI service with `/predict`, `/models`, `/forecast/full`, `/export`  
- Streamlit operations dashboard (dark navy / teal theme)  
- SHAP plots, Plotly reports, CSV export  

## Tech stack

| Area | Stack |
|------|--------|
| Language | Python 3.10+ |
| API | FastAPI, Uvicorn, Pydantic v2 |
| UI | Streamlit ≥ 1.32, Plotly |
| ML | scikit-learn, XGBoost, Prophet, pmdarima (optional), SHAP |
| Data | pandas, NumPy |

## Folder structure

```
├── data/ferry_tickets.csv
├── notebooks/01_eda_and_modeling.ipynb
├── scripts/train_all.py
├── backend/
│   ├── main.py
│   ├── routers/
│   ├── services/
│   └── utils/
├── frontend/
│   ├── app.py
│   └── pages/
├── models/                # .pkl, metadata.json, metrics_comparison.csv
├── reports/figures/       # HTML/PNG charts
├── logs/
├── tests/
├── requirements.txt
├── requirements-training.txt
├── packages.txt
├── Procfile
├── render.yaml
└── README.md
```

## Setup

```bash
cd "/path/to/Short-Term Ferry Ticket Demand Forecasting & Predictive Decision Support System"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-training.txt
# macOS XGBoost may require: brew install libomp
cp .env.example .env
# Ensure data path exists (already copied to data/ferry_tickets.csv in this repo)
export PYTHONPATH=.
# Optional fast training:
export FERRY_TRAIN_MAX_ROWS=40000
export SKIP_SARIMA=1   # if pmdarima import fails on your platform
python scripts/train_all.py
```

**Streamlit Community Cloud** uses the slim `requirements.txt` only (Prophet/pmdarima/SHAP are training-only and often break the cloud installer). `packages.txt` installs `libgomp1` for XGBoost on Linux. Train models locally; `.pkl` files stay gitignored by default.

Run API:

```bash
export PYTHONPATH=.
uvicorn backend.main:app --reload
```

Run Streamlit:

```bash
streamlit run frontend/app.py
```

**Config (no `st.secrets` banner locally):** The app reads `API_URL` and optional `APP_PASSWORD` from **environment variables** first, then from **`.streamlit/secrets.toml`** (if you create it). It does **not** use `st.secrets`, so you will not see Streamlit’s “No secrets found” message when that file is missing. For Streamlit Community Cloud, add the same keys in **App settings → Secrets** and copy them into a local `secrets.toml` only for local testing; or set `API_URL` in the deploy environment if your host supports it.

## Screenshots

- `docs/screenshots/dashboard.png` (placeholder — add after UI capture)  
- `docs/screenshots/models.png`  

## Model performance (from `models/metrics_comparison.csv`)

After training, open `models/metrics_comparison.csv` for the full table. Example rows (subset):

| target | horizon | model | mae | rmse | mape |
|--------|---------|-------|-----|------|------|
| sales | 15min | gradient_boosting | 7.25 | 15.03 | 149.26 |
| sales | 1hr | random_forest | (see CSV) | | |
| redemption | 1hr | xgboost | (see CSV) | | |

SARIMA/Prophet may be empty if strided fitting did not converge or was skipped.

## License

MIT License — see [LICENSE](LICENSE).
