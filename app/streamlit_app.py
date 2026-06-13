"""Streamlit app for the Insiders loyalty segmentation model.

Two views, both served from the serialized pipeline (never retrains) and the small
scored base in ``data/sample/`` (never reads ``raw/`` or ``processed/``):
  * Customer base — the full scored population, segment-coloured, downloadable.
  * Classify a customer — on-demand, synchronous single-customer scoring.
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

st.set_page_config(page_title="Insiders Loyalty Segmentation", page_icon="🎯", layout="wide")

# RdYlGn ramp ordered best (green) -> worst (red); indexed by value rank.
_RAMP = [
    "#006837", "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#ffffbf",
    "#fee08b", "#fdae61", "#f46d43", "#d73027", "#a50026",
]


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_base() -> pd.DataFrame:
    return pd.read_parquet(config.SAMPLE_FILE)


@st.cache_data
def load_card() -> dict:
    return json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _rank_color(rank: int, n: int) -> str:
    idx = 0 if n <= 1 else round(rank / (n - 1) * (len(_RAMP) - 1))
    return _RAMP[idx]


def _white_to_green(p: float) -> str:
    r, g, b = (int(255 + (target - 255) * p) for target in (26, 152, 80))
    return f"#{r:02x}{g:02x}{b:02x}"


try:
    predictor = load_predictor()
    base = load_base()
    card = load_card()
except FileNotFoundError:
    st.error("Trained model not found. Run `python -m src.pipeline` to generate the pipeline first.")
    st.stop()

segment_names = {int(k): v for k, v in card["segment_names"].items()}
name_to_rank = {v: int(k) for k, v in card["segment_names"].items()}
n_segments = len(segment_names)
business = card.get("business", {})

st.title("🎯 Insiders Loyalty Segmentation")
st.caption("RFM + K-Means segmentation of an e-commerce customer base. Model loaded from a serialized pipeline.")
st.info(
    f"**Insiders** are {business.get('insiders_pct_customers', 0)}% of customers but hold "
    f"{business.get('insiders_pct_revenue', 0)}% of revenue "
    f"({business.get('insiders_revenue_lift', 0)}× revenue concentration)."
)

base_tab, predict_tab = st.tabs(["📊 Customer base", "🎯 Classify a customer"])

# --------------------------------------------------------------------------- #
# Tab 1 — segmented customer base
# --------------------------------------------------------------------------- #
with base_tab:
    total_revenue = float(base["monetary"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(base):,}")
    c2.metric("Total revenue", f"£{total_revenue:,.0f}")
    c3.metric("Insiders", f"{int(business.get('insiders_customers', 0)):,}")
    c4.metric("Insiders revenue", f"{business.get('insiders_pct_revenue', 0)}%")

    st.markdown("**Segments** (ranked by value)")
    summary = pd.DataFrame(
        [
            {
                "Segment": seg["name"],
                "Customers": int(seg["n_customers"]),
                "% Customers": seg["pct_customers"],
                "% Revenue": seg["pct_revenue"],
                "Avg Monetary (£)": seg["monetary"],
            }
            for _, seg in sorted(card["segments"].items(), key=lambda kv: int(kv[0]))
        ]
    )

    def _color_segment(col: pd.Series) -> list[str]:
        styles = []
        for value in col:
            bg = _rank_color(name_to_rank.get(value, n_segments - 1), n_segments)
            fg = "#000000" if _luminance(bg) > 0.6 else "#ffffff"
            styles.append(f"background-color: {bg}; color: {fg}")
        return styles

    st.dataframe(
        summary.style.apply(_color_segment, subset=["Segment"]).format(
            {"% Customers": "{:.1f}", "% Revenue": "{:.1f}", "Avg Monetary (£)": "{:,.0f}", "Customers": "{:,}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Customers** — coloured by segment value, graded by revenue")
    options = ["All segments"] + [segment_names[r] for r in sorted(segment_names)]
    choice = st.selectbox("Filter", options, index=0)

    view = base if choice == "All segments" else base[base["segment"] == choice]
    display = (
        view.sort_values("monetary", ascending=False)
        .loc[:, ["customer_id", "recency_days", "frequency", "monetary", "avg_ticket", "total_items", "segment"]]
        .rename(
            columns={
                "customer_id": "Customer",
                "recency_days": "Recency (d)",
                "frequency": "Frequency",
                "monetary": "Monetary (£)",
                "avg_ticket": "Avg Ticket (£)",
                "total_items": "Items",
                "segment": "Segment",
            }
        )
    )

    def _grade_monetary(col: pd.Series) -> list[str]:
        pct = col.rank(pct=True)
        return [f"background-color: {_white_to_green(p)}" for p in pct]

    st.dataframe(
        display.style.apply(_color_segment, subset=["Segment"])
        .apply(_grade_monetary, subset=["Monetary (£)"])
        .format(
            {
                "Customer": "{:.0f}",
                "Recency (d)": "{:.0f}",
                "Frequency": "{:.0f}",
                "Monetary (£)": "{:,.0f}",
                "Avg Ticket (£)": "{:,.2f}",
                "Items": "{:,.0f}",
            }
        ),
        use_container_width=True,
        height=460,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download segmented list (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="insiders_customer_segments.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Tab 2 — single-customer scoring
# --------------------------------------------------------------------------- #
with predict_tab:

    def _number_input(label: str, series: pd.Series, *, integer: bool) -> float:
        median = float(series.median())
        hi = float(series.quantile(0.99)) * 2
        if integer:
            return float(st.number_input(label, min_value=0, max_value=int(hi) or 1, value=int(median), step=1))
        return float(st.number_input(label, min_value=0.0, max_value=round(hi, 2), value=round(median, 2), step=10.0))

    st.markdown("Enter a customer's RFM profile to assign a segment on demand.")
    left, right = st.columns(2)
    with left:
        recency = _number_input("Recency (days since last purchase)", base["recency_days"], integer=True)
        frequency = _number_input("Frequency (number of orders)", base["frequency"], integer=True)
        monetary = _number_input("Monetary (total spend)", base["monetary"], integer=False)
    with right:
        avg_ticket = _number_input("Average ticket (spend per line)", base["avg_ticket"], integer=False)
        total_items = _number_input("Total items purchased", base["total_items"], integer=True)

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
        m1, m2 = st.columns(2)
        m1.metric("Segment", result.segment)
        m2.metric("Assignment confidence", f"{result.confidence * 100:.1f}%")
        st.success(f"This customer is classified as **{result.segment}** — {badge}.")

        segment_stats = card.get("segments", {}).get(str(result.cluster_id), {})
        if segment_stats:
            st.markdown("**Segment profile vs. this customer**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "This customer": [recency, frequency, monetary, avg_ticket, total_items],
                        "Segment average": [segment_stats.get(f, float("nan")) for f in config.RFM_FEATURES],
                    },
                    index=config.RFM_FEATURES,
                ).round(2),
                use_container_width=True,
            )

        st.markdown("**What defines this segment** (standardized distance from the average customer)")
        st.bar_chart(
            pd.DataFrame(
                {"segment profile (z)": [fc.segment_zscore for fc in result.feature_contributions]},
                index=[fc.feature for fc in result.feature_contributions],
            )
        )
        top = result.feature_contributions[0]
        direction = "above" if top.segment_zscore > 0 else "below"
        st.caption(f"Most distinctive trait: **{top.feature}** is well {direction} average for this segment.")
