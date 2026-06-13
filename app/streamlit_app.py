"""Streamlit app: score a customer into an RFM segment on demand.

Loads the serialized pipeline (never retrains) and uses ``data/sample/`` only for
input ranges and population context. Prediction is synchronous per user request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor, RFMInput  # noqa: E402

st.set_page_config(page_title="Insiders Loyalty Segmentation", page_icon="🎯", layout="centered")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_parquet(config.SAMPLE_FILE)


@st.cache_data
def load_card() -> dict:
    return json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))


def _number_input(label: str, series: pd.Series, *, integer: bool) -> float:
    median = float(series.median())
    hi = float(series.quantile(0.99)) * 2
    if integer:
        return float(st.number_input(label, min_value=0, max_value=int(hi) or 1, value=int(median), step=1))
    return float(st.number_input(label, min_value=0.0, max_value=round(hi, 2), value=round(median, 2), step=10.0))


st.title("🎯 Insiders Loyalty Segmentation")
st.caption("Score a customer's purchasing behaviour into a value-ranked segment.")

try:
    predictor = load_predictor()
    sample = load_sample()
    card = load_card()
except FileNotFoundError:
    st.error("Trained model not found. Run `python -m src.pipeline` to generate the pipeline first.")
    st.stop()

business = card.get("business", {})
st.info(
    f"**Insiders** are {business.get('insiders_pct_customers', 0)}% of customers but hold "
    f"{business.get('insiders_pct_revenue', 0)}% of revenue "
    f"({business.get('insiders_revenue_lift', 0)}× revenue concentration)."
)

st.subheader("Customer RFM inputs")
left, right = st.columns(2)
with left:
    recency = _number_input("Recency (days since last purchase)", sample["recency_days"], integer=True)
    frequency = _number_input("Frequency (number of orders)", sample["frequency"], integer=True)
    monetary = _number_input("Monetary (total spend)", sample["monetary"], integer=False)
with right:
    avg_ticket = _number_input("Average ticket (spend per line)", sample["avg_ticket"], integer=False)
    total_items = _number_input("Total items purchased", sample["total_items"], integer=True)

if st.button("Classify customer", type="primary"):
    try:
        result = predictor.predict(
            RFMInput(
                recency_days=recency,
                frequency=frequency,
                monetary=monetary,
                avg_ticket=avg_ticket,
                total_items=total_items,
            )
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    badge = "⭐ Insider" if result.is_insider else "Standard segment"
    c1, c2 = st.columns(2)
    c1.metric("Segment", result.segment)
    c2.metric("Assignment confidence", f"{result.confidence * 100:.1f}%")
    st.success(f"This customer is classified as **{result.segment}** — {badge}.")

    segment_stats = card.get("segments", {}).get(str(result.cluster_id), {})
    if segment_stats:
        st.markdown("**Segment profile vs. this customer**")
        comparison = pd.DataFrame(
            {
                "This customer": [recency, frequency, monetary, avg_ticket, total_items],
                "Segment average": [segment_stats.get(f, float("nan")) for f in config.RFM_FEATURES],
            },
            index=config.RFM_FEATURES,
        ).round(2)
        st.dataframe(comparison, use_container_width=True)

    st.markdown("**What defines this segment** (standardized distance from the average customer)")
    drivers = pd.DataFrame(
        {"segment profile (z)": [fc.segment_zscore for fc in result.feature_contributions]},
        index=[fc.feature for fc in result.feature_contributions],
    )
    st.bar_chart(drivers)
    top = result.feature_contributions[0]
    direction = "above" if top.segment_zscore > 0 else "below"
    st.caption(
        f"Most distinctive trait: **{top.feature}** is well {direction} average for this segment."
    )
