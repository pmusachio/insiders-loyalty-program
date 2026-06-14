# Insiders Loyalty Program — RFM Customer Segmentation

> Unsupervised learning · RFM features · KMeans clustering · Value-based segmentation

## Business Problem

An e-commerce company wants a loyalty program ("Insiders") that concentrates its budget on the
customers who actually drive revenue. The question is which customers belong in that high-value
tier, and how the rest of the base should be grouped for differentiated treatment.

There is no label to predict, so this is an **unsupervised segmentation** problem. The cost of a
poor segmentation is misallocated loyalty spend — perks given to low-value customers, while the
true top tier is under-served. The segmentation is judged on cluster quality and, decisively, on
whether one segment captures a disproportionate share of revenue that justifies a dedicated program.

## Dataset

[E-Commerce Data (UCI Online Retail)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

| Property | Value |
|----------|-------|
| Rows | ~540k transaction lines |
| Customers (after cleaning) | 4,338 |
| Fields | invoice, stock code, quantity, unit price, invoice date, customer id, country |
| Target | none (unsupervised) |

## Solution Strategy

1. **Acquisition** — pull the dataset from Kaggle on demand; a versioned scored sample backs an offline run.
2. **Cleaning** — drop cancellations and non-positive quantities/prices, require a customer id.
3. **RFM engineering** — aggregate transactions to one row per customer: recency, frequency, monetary, average ticket and total items.
4. **Preparation** — log-transform the skewed RFM features and standardize, inside the model `Pipeline` so serving reuses the exact transform.
5. **Model selection** — KMeans is fit across a range of k and the number of segments is chosen on clustering quality, then segments are ranked and named by value.
6. **Activation** — the top segment is profiled as the Insiders tier with its revenue contribution quantified.

## Top Insights & Hypotheses

- **A small tier drives most revenue.** The Insiders segment is 16% of customers but **64% of revenue** — a textbook case for a dedicated program.
- **Revenue concentration is steep.** Insiders spend on average 3.9x the overall customer mean.
- **Recency separates value sharply** — Insiders ordered ~17 days ago on average versus ~160 days for the At-Risk tier.
- **The largest segments are the least valuable** (Loyal Customers 37% / 21% of revenue, At-Risk 39% / 6%), so headcount is a poor proxy for value.

## Engineered Features

| Feature | Definition | Business signal |
|---------|-----------|-----------------|
| recency_days | days since the customer's last purchase | engagement freshness |
| frequency | number of distinct invoices | purchase cadence |
| monetary | total gross revenue from the customer | direct value |
| avg_ticket | mean revenue per invoice | basket size |
| total_items | total quantity purchased | volume |

## Model

KMeans on log-transformed, standardized RFM features, with k chosen by clustering quality and the
full preprocessing-plus-KMeans pipeline serialized for skew-free serving.

| Metric | Value |
|--------|------:|
| Selected k | 4 |
| Silhouette | 0.269 |
| Davies-Bouldin | 1.173 |
| Calinski-Harabasz | 1912 |

## Business Results

| Segment | Customers % | Revenue % | Avg recency (days) | Avg frequency | Avg monetary |
|---------|------------:|----------:|-------------------:|--------------:|-------------:|
| Insiders | 16.5% | 64.5% | 17 | 13.8 | 8,046 |
| Loyal Customers | 37.0% | 21.3% | 50 | 3.6 | 1,180 |
| Promising | 7.3% | 8.7% | 125 | 2.2 | 2,454 |
| At Risk | 39.2% | 5.6% | 160 | 1.3 | 291 |

The Insiders tier — 714 customers generating **$5.7M, 64% of revenue at 3.9x the average customer
value** — is the program's clear focus, while At-Risk customers are a separate win-back problem.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/insiders-loyalty-program.git
   cd insiders-loyalty-program
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [huggingface.co/spaces/pmusachio/insiders-loyalty-program](https://huggingface.co/spaces/pmusachio/insiders-loyalty-program) — profile a customer and see their loyalty segment.

## Next Steps

- Validate the tiers against forward revenue: the real test is whether Insiders-targeted perks lift retention and spend versus a control.
- Add product-category and tenure features to split the large low-value segments into actionable win-back groups.
- Re-score on a schedule, since RFM position drifts as customers buy or lapse.
