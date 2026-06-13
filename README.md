# **Insiders Loyalty Program — RFM Customer Segmentation**

> Unsupervised Learning · Customer Segmentation · K-Means · Production Pipeline · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Business Problem

A UK-based online retailer wants to launch **Insiders**, an exclusive loyalty tier
with premium perks. The cost structure is asymmetric:

- **Inviting a low-value customer** wastes perk budget (discounts, early access, support cost).
- **Missing a genuinely high-value customer** forfeits retention of the revenue that
  actually sustains the business.

The decision the model informs is concrete: **which customers to enrol in Insiders**,
out of thousands, repeatably and without manual cherry-picking.

**Why a model and not just a rule?** Classic RFM scoring (rank customers into
quantiles) is a perfectly reasonable heuristic and is used here as the **baseline**.
We adopt K-Means only because it must *earn its complexity* — it produces segments
that are ~3× more internally cohesive (silhouette 0.27 vs 0.09) without arbitrary
quantile cut-offs, and a tighter, more defensible Insiders definition. See
[Model](#model).

**Assumptions:** behaviour is summarised by RFM; "value" is unlabelled and inferred
from purchasing patterns; recency is measured against a fixed snapshot date.

---

## Dataset

**Source:** [E-Commerce Data — Kaggle (UCI Online Retail)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

| Property | Value |
|---|---|
| Granularity | One row per transaction line |
| Raw rows | 541,909 transactions |
| Customers (after cleaning) | 4,338 |
| Raw columns | `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country` |

The raw CSV (~40 MB) is **not committed**; it is fetched on demand via the Kaggle API.

---

## Solution Strategy

1. **Data acquisition** — download via Kaggle API into `data/raw/` (idempotent; skipped if present).
2. **Cleaning** — drop missing customers, parse dates, keep positive quantity/price,
   remove cancelled invoices (`InvoiceNo` prefixed `C`).
3. **Feature engineering** — aggregate transactions into one RFM row per customer.
4. **Preprocessing pipeline** (sklearn `ColumnTransformer`): `SimpleImputer(median) → log1p → StandardScaler`.
   `log1p` compresses the right-skewed RFM tails before distance-based clustering.
5. **Data leakage** — structurally not applicable (unsupervised, no target). The only
   temporal dependency, `recency_days`, is computed against a snapshot reference date
   (`max(InvoiceDate) + 1 day`); production scoring must recompute RFM as-of the scoring date.
6. **Class balancing** — not applicable (no classes). Uneven segment sizes are
   expected and desirable — Insiders are meant to be a minority.
7. **Baseline → model** — rule-based RFM scoring, then K-Means with a grid search over `k`.
8. **Validation** — stability via bootstrap resampling (Adjusted Rand Index), the
   clustering analogue of cross-validation.
9. **Evaluation** — internal indices + per-segment profiling + boundary (error) analysis.
10. **Serving** — serialize the full pipeline (`models/pipeline.joblib`) and expose an
    on-demand Streamlit app. The transform is bundled with the model — no training-serving skew.

---

## Top Insights & Hypotheses

- **Revenue is highly concentrated.** The top segment (16.5% of customers) generates
  **64.5% of revenue** — a 3.9× concentration. This is the core justification for Insiders.
- **Insiders are structurally different**, not just bigger spenders: they order ~14×
  per year (vs 1.3× for the dormant majority) with 17-day recency (vs 160 days).
- **A high-ticket, low-frequency niche exists** (the *Promising* segment: £671 average
  ticket, ~2 orders, 124-day recency) that behaves like wholesale buyers. It is the
  least cohesive group (27% boundary share) and warrants separate treatment.
- **~39% of customers are dormant/at-risk** (160-day recency) yet contribute only 5.6%
  of revenue — low priority for premium perks, candidates for win-back instead.

---

## Engineered Features

All features are domain aggregations from raw transactions (the *engineered* layer);
the `log1p` + scaling parameters and centroids are *learned* inside the pipeline.

| Feature | Formula | Business signal |
|---|---|---|
| `recency_days` | `(max(InvoiceDate)+1d − last purchase).days` | Engagement freshness; high = churn risk |
| `frequency` | count of distinct invoices | Purchase habit / loyalty |
| `monetary` | Σ(`Quantity` × `UnitPrice`) | Total revenue contribution |
| `avg_ticket` | mean(`Quantity` × `UnitPrice`) per line | Basket value / price tier |
| `total_items` | Σ `Quantity` | Volume of consumption |

---

## Model

**Baseline vs. final model**

| Approach | Method | Top-segment size | Top-segment revenue | Silhouette |
|---|---|---|---|---|
| Baseline | RFM quintile scoring (heuristic) | 21.4% | 70.1% | 0.091 |
| **Final** | **K-Means (k=4), log-scaled RFM** | **16.5% (Insiders)** | **64.5%** | **0.269** |

The heuristic captures slightly more raw revenue, but in a looser, larger bucket.
K-Means trades a little revenue coverage for **3× better cohesion** and a sharper
Insiders definition.

**Model selection** — grid search over `k`, scored by three internal indices:

| k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| 3 | 0.256 | 1.295 | 2200.7 |
| **4** | **0.269** | **1.173** | 1912.3 |
| 5 | 0.231 | 1.286 | 1729.3 |
| 6 | 0.216 | 1.266 | 1562.0 |
| 7 | 0.200 | 1.316 | 1442.3 |
| 8 | 0.207 | 1.254 | 1461.0 |
| 9 | 0.200 | 1.271 | 1376.4 |

`k=4` wins on silhouette **and** Davies-Bouldin. (Calinski-Harabasz prefers fewer
clusters monotonically, so it is used as a tie-checker, not the selector.)

**Final metrics**

| Metric | Value | Reading |
|---|---|---|
| Silhouette | 0.269 | Moderate but real separation — expected for continuous behavioural data |
| Davies-Bouldin | 1.173 | Lower is better |
| Stability (mean ARI) | 0.678 ± 0.196 | Assignments are reasonably reproducible across resamples |
| Boundary share | 3.25% | Few customers sit ambiguously between segments |

---

## Business Results

| Cluster | Segment | Customers | % Customers | % Revenue | Recency (d) | Frequency | Avg Monetary (£) |
|---|---|---|---|---|---|---|---|
| 0 | **Insiders** | 714 | 16.5% | **64.5%** | 17 | 13.8 | 8,046 |
| 2 | Loyal Customers | 1,607 | 37.0% | 21.3% | 50 | 3.6 | 1,180 |
| 1 | Promising | 316 | 7.3% | 8.7% | 125 | 2.2 | 2,454 |
| 3 | At Risk | 1,701 | 39.2% | 5.6% | 160 | 1.3 | 291 |

**ML → business translation:** targeting the **714 Insiders** (16.5% of the base)
covers **£5.74M / 64.5%** of revenue at a **3.9× concentration lift**. A retention
campaign focused on this segment protects the majority of revenue while spending
perk budget on the smallest possible audience — the explicit cost trade-off from the
[Business Problem](#business-problem).

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/pmusachio/insiders-loyalty-program.git
cd insiders-loyalty-program

# 2. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Kaggle credentials
#    Place your kaggle.json at ~/.kaggle/kaggle.json (Kaggle → Settings → API).

# 4. Run the full pipeline (download → preprocess → baseline → train → evaluate → serialize)
python -m src.pipeline

# 5. Tests
pytest tests/

# 6. App (local)
streamlit run app/streamlit_app.py
```

**Live demo:** try the interactive segmenter on Hugging Face Spaces →
`https://huggingface.co/spaces/<your-space>` <!-- replace with your published Space URL -->

---

## Next Steps

- **Drift & performance monitoring.** Track the RFM feature distributions and segment
  sizes monthly; alert when the population shifts (e.g., PSI on each feature) or when
  the Insiders share drifts materially from ~16%.
- **Retraining triggers.** Refit when stability (ARI) on a fresh snapshot degrades, when
  silhouette drops below ~0.22, or on a fixed monthly cadence — whichever comes first.
- **Known limitations (conscious trade-offs):**
  - Silhouette ≈ 0.27 reflects genuinely continuous behaviour; segments are useful but
    not crisply separated. A soft-assignment model (e.g., Gaussian Mixture) is a
    candidate if probabilistic membership becomes a requirement.
  - RFM features are correlated (`monetary`, `avg_ticket`, `total_items`); K-Means on
    Euclidean distance tolerates this, but a dimensionality step is a future option.
  - `recency` is snapshot-relative; a rolling reference date is needed for live scoring.
  - **No Git LFS / feature store / model registry** — deliberately omitted. No artifact
    exceeds 50 MB and the model is a 37 KB joblib; that infrastructure would be
    disproportionate to the project's scale.
