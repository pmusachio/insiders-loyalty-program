"""Interactive RFM customer-segmentation dashboard.

Scores a customer's recency/frequency/monetary profile into a value-ranked loyalty
segment and shows where they fall on the customer map, with the segment overview.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor, RFMInput  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Insiders Loyalty Segmentation", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(config.SAMPLE_FILE) if config.SAMPLE_FILE.exists() else pd.DataFrame()


@st.cache_data
def load_card() -> dict:
    return json.loads(config.MODEL_CARD_PATH.read_text()) if config.MODEL_CARD_PATH.exists() else {}


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def customer_map(df, point, point_segment, names):
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=D["background"])
    segs = list(names.values()) if names else sorted(df["segment"].unique())
    for i, seg in enumerate(segs):
        m = df["segment"] == seg
        ax.scatter(df.loc[m, "frequency"], np.log1p(df.loc[m, "monetary"]), s=9, alpha=0.5,
                   color=config.SEGMENT_COLORS[i % len(config.SEGMENT_COLORS)], label=seg)
    ax.scatter([point[0]], [np.log1p(point[1])], s=240, marker="*", color=D["foreground"],
               edgecolors=D["purple"], linewidths=2, zorder=5, label="This customer")
    ax.set_xlabel("Frequency (orders)")
    ax.set_ylabel("Monetary (log)")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=7)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    card = load_card()
    sample = load_sample()
    names = {int(k): v for k, v in card.get("segment_names", {}).items()}

    st.title("Insiders Loyalty Program — RFM Segmentation")
    st.markdown(
        "Scores a customer's purchase behaviour into a value-ranked loyalty segment to target the "
        "loyalty program where it pays off most. Built with RFM features and KMeans."
    )

    with st.sidebar:
        st.header("Customer (RFM)")
        recency = st.slider("Recency (days since last order)", 0, 380, 30)
        frequency = st.slider("Frequency (orders)", 1, 100, 8)
        monetary = st.number_input("Monetary (total spend)", 0.0, 100000.0, 4000.0, 100.0)
        avg_ticket = st.number_input("Average ticket", 0.0, 5000.0, 50.0, 5.0)
        total_items = st.number_input("Total items", 0.0, 100000.0, 1500.0, 50.0)
        run = st.button("Assign segment", type="primary")

    if run:
        pred = predictor.predict(RFMInput(recency, frequency, monetary, avg_ticket, total_items))
        st.subheader("Assigned segment")
        c = st.columns(3)
        c[0].metric("Segment", pred.segment)
        c[1].metric("Insider", "Yes" if pred.is_insider else "No")
        c[2].metric("Confidence", f"{pred.confidence*100:.0f}%")
        color = D["green"] if pred.is_insider else D["foreground"]
        st.markdown(
            f"<span style='color:{color}'>This customer is assigned to the "
            f"<b>{pred.segment}</b> segment.</span>", unsafe_allow_html=True)
        if not sample.empty:
            st.pyplot(customer_map(sample, (frequency, monetary), pred.cluster_id, names))

    if card.get("segments"):
        st.subheader("Segments overview")
        rows = []
        for k, s in sorted(card["segments"].items(), key=lambda kv: int(kv[0])):
            rows.append({"Segment": s["name"], "Customers %": round(s["pct_customers"], 1),
                         "Revenue %": round(s["pct_revenue"], 1),
                         "Avg recency (d)": round(s["recency_days"], 0),
                         "Avg frequency": round(s["frequency"], 1),
                         "Avg monetary": round(s["monetary"], 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
