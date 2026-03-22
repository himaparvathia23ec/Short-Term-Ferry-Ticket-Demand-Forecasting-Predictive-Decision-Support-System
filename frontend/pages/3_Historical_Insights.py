"""Historical insights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    p = ROOT / "data" / "ferry_tickets.csv"
    df = pd.read_csv(p)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("Timestamp")


st.title("Historical insights")
df = load_df()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Sales Count"], name="Sales", line=dict(color="#00b4d8")))
fig.add_trace(go.Scatter(x=df["Timestamp"], y=df["Redemption Count"], name="Redemptions", line=dict(color="#7ee787")))
fig.update_layout(template="plotly_dark", title="Toronto Island Ferry — Demand Intelligence — Full history", height=500)
st.plotly_chart(fig, use_container_width=True)

tmp = df.copy()
tmp["hour"] = tmp["Timestamp"].dt.hour
tmp["dow"] = tmp["Timestamp"].dt.day_name()
heat = tmp.groupby(["dow", "hour"], observed=False)["Sales Count"].mean().reset_index()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat["dow"] = pd.Categorical(heat["dow"], categories=order, ordered=True)
pivot = heat.pivot(index="dow", columns="hour", values="Sales Count")
fig2 = px.imshow(pivot, aspect="auto", color_continuous_scale="Teal", title="Avg sales — hour × weekday")
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

monthly = df.set_index("Timestamp").resample("ME")["Sales Count"].mean().reset_index()
fig3 = px.bar(monthly, x="Timestamp", y="Sales Count", title="Monthly average sales (interval mean)")
fig3.update_layout(template="plotly_dark")
st.plotly_chart(fig3, use_container_width=True)

top = df.nlargest(20, "Sales Count")[["Timestamp", "Sales Count", "Redemption Count"]]
st.subheader("Top 20 highest-sales intervals")
st.dataframe(top, use_container_width=True)
