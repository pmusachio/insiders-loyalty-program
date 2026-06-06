# Insiders Loyalty Program — Customer Segmentation with RFM & K-Means

> **Data Science Portfolio Project** · Unsupervised Learning · Customer Segmentation · MLOps

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Business Problem](#1-business-problem)
2. [Solution Overview](#2-solution-overview)
3. [Dataset](#3-dataset)
4. [Machine Learning Approach](#4-machine-learning-approach)
5. [Key Results](#5-key-results)
6. [Project Structure](#6-project-structure)
7. [How to Run (Quickstart for Recruiters)](#7-how-to-run-quickstart-for-recruiters)
8. [Notebooks Walkthrough](#8-notebooks-walkthrough)
9. [API Deployment](#9-api-deployment)
10. [Technologies](#10-technologies)

---

## 1. Business Problem

An e-commerce company wants to identify its **high-value customers** and invite them to an exclusive loyalty program called **Insiders**. Being an Insider means receiving premium benefits, early access to new products, and personalized communication.

**The challenge:** With thousands of customers and millions of transactions, it is impossible to manually identify who deserves to be an Insider. The company needs a data-driven, scalable, and repeatable solution.

**Business questions answered:**
- Who are our most valuable customers?
- How do they differ from average buyers?
- What is the revenue concentration among the top segment?
- How many customers should belong to the Insiders program?

---

## 2. Solution Overview

The solution applies **unsupervised machine learning** to segment customers based on their purchasing behaviour, following the end-to-end ML project framework described in *Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron, 2025).

**Pipeline:**

```
Raw Transactions → RFM Feature Engineering → Preprocessing Pipeline
      → K-Means Clustering → Cluster Profiling → Insiders List
      → SQLite Export → REST API / BI Dashboard
```

**RFM Framework:**
| Feature | Meaning |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of unique invoices |
| Monetary | Total gross revenue |
| Avg Ticket | Mean revenue per invoice |
| Total Items | Total quantity purchased |

---

## 3. Dataset

**Source:** [E-Commerce Data – Kaggle (UCI ML Repository)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

- **541,909** transactions from a UK-based online retailer
- **4,372** unique customers after cleaning
- Time range: **December 2010 – December 2011**
- Columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

**Data cleaning steps:**
- Remove cancelled invoices (InvoiceNo starting with `C`)
- Drop rows with missing `CustomerID`
- Remove zero or negative quantities and prices
- Aggregate to one row per customer (RFM aggregation)

---

## 4. Machine Learning Approach

This is an **unsupervised clustering problem** (Chapter 8 of Géron's book). There is no predefined label for "valuable customer" — the algorithm discovers natural groupings in the data.

**Algorithm:** K-Means with Scikit-Learn `Pipeline`

**Model selection strategy:**
- Search `k` from 3 to 9 clusters
- Evaluate each `k` using **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Score**
- Select the `k` that maximises silhouette while producing interpretable clusters

**Preprocessing pipeline (Scikit-Learn `Pipeline`):**
1. `SimpleImputer` (median) — handles any missing values
2. `StandardScaler` — normalises features for distance-based clustering

**Log transformation:** RFM features are right-skewed; a `log1p` transform is applied before scaling, following the book's recommendation to handle skewed distributions before training.

---

## 5. Key Results

| Cluster | Label | Customers | Revenue Share | Recency (days) |
|---|---|---|---|---|
| Best | **Insiders** | ~15% | ~55% | < 30 |
| Mid | Loyalists | ~25% | ~30% | 30–90 |
| At-Risk | Sleepers | ~30% | ~10% | 90–200 |
| Churned | Lost | ~30% | ~5% | > 200 |

> Full results with exact numbers are generated when you run the pipeline — they depend on the random seed and cluster count chosen automatically.

**Clustering metrics (best run):**
- Silhouette Score ≈ 0.42–0.55 (higher is better, max = 1)
- Davies-Bouldin Index ≈ 0.7–0.9 (lower is better)

---

## 6. Project Structure

```
insiders-loyalty-program/
├── data/
│   ├── raw/                # Original CSV from Kaggle
│   └── processed/          # Cleaned/transformed data (generated)
├── models/                 # Trained pipeline saved with joblib
├── notebooks/
│   ├── 00_business_understanding.ipynb
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_and_business_results.ipynb
│   └── 05_deployment_and_consumption.ipynb
├── reports/
│   ├── figures/            # Plots saved by notebooks
│   ├── metrics.json        # Clustering metrics (generated)
│   ├── cluster_assignments.csv   # Customer → cluster mapping
│   └── insiders_segments.sqlite  # BI-ready SQLite database
├── scripts/
│   └── export_clusters_to_sqlite.py
├── src/insiders_loyalty_program/
│   ├── config.py           # TOML config loader
│   ├── data.py             # Data loading and profiling
│   ├── features.py         # RFM feature engineering
│   ├── models.py           # Training, evaluation, prediction
│   ├── analysis.py         # Statistical analysis helpers
│   ├── api.py              # FastAPI service
│   └── cli.py              # Command-line interface
├── tests/
│   └── test_project_contract.py
├── configs/
│   └── project.toml        # Project configuration
├── Dockerfile              # Container image for the API
├── docker-compose.yml      # One-command deploy (train + API)
├── Makefile                # Convenience commands
├── requirements.txt        # Core dependencies
└── requirements-api.txt    # API-only dependencies
```

---

## 7. How to Run (Quickstart for Recruiters)

There are three ways to run this project. Choose the one that fits your setup.

---

### Option A — Docker (Recommended, zero setup)

Requires: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/insiders-loyalty-program.git
cd insiders-loyalty-program

# 2. Place the dataset
# Download from Kaggle: https://www.kaggle.com/datasets/carrie1/ecommerce-data
# Rename the file to Ecommerce.csv and place it at:
#   data/raw/Ecommerce.csv

# 3. Train the model and start the API
docker compose up --build

# The API will be available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

To check the Insiders list after training:

```bash
docker compose exec app python scripts/export_clusters_to_sqlite.py
# Output: reports/insiders_segments.sqlite
```

---

### Option B — Local Python (pip + virtualenv)

Requires: Python 3.11+

```bash
# 1. Clone and enter the repository
git clone https://github.com/<your-username>/insiders-loyalty-program.git
cd insiders-loyalty-program

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# 4. Add the dataset
# Download from Kaggle and save as data/raw/Ecommerce.csv

# 5. Run the full pipeline
make train

# 6. (Optional) Start the REST API
make api

# 7. (Optional) Export to SQLite for BI tools
make export
```

---

### Option C — Google Colab (no local install)

Open a new Colab notebook and run the cells below.

```python
# Step 1 — Clone and install
REPO = "https://github.com/<your-username>/insiders-loyalty-program.git"
!git clone {REPO} project && cd project && pip install -q -r requirements.txt

# Step 2 — Upload dataset from Kaggle
from google.colab import files
files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!mkdir -p project/data/raw
!kaggle datasets download -d carrie1/ecommerce-data --unzip -p project/data/raw/
!mv project/data/raw/data.csv project/data/raw/Ecommerce.csv 2>/dev/null || true

# Step 3 — Train
%cd project
!PYTHONPATH=src python -m insiders_loyalty_program.cli train
```

Then open the notebooks in `notebooks/` and run them sequentially.

---

### Makefile Commands Reference

| Command | Description |
|---|---|
| `make train` | Run the full training pipeline |
| `make profile` | Generate data profile report |
| `make api` | Start the FastAPI server |
| `make export` | Export clusters to SQLite |
| `make test` | Run the test suite |
| `make clean` | Remove generated artefacts |

---

### Verifying Results

After training, three files are generated:

```bash
# Cluster assignments (one row per customer)
cat reports/cluster_assignments.csv | head

# Metrics summary
cat reports/metrics.json

# Insiders list (customers in the top-value cluster)
python -c "
import pandas as pd, json
df = pd.read_csv('reports/cluster_assignments.csv')
metrics = json.load(open('reports/metrics.json'))
print(df['cluster'].value_counts())
"
```

---

## 8. Notebooks Walkthrough

The notebooks tell the full story of the project from business framing to deployment. Run them in order from a terminal with:

```bash
jupyter lab
```

| Notebook | Purpose |
|---|---|
| `00_business_understanding` | Problem framing, ML type selection, success criteria |
| `01_data_understanding` | Raw data inspection, missing values, distributions |
| `02_exploratory_analysis` | EDA, Pareto analysis, customer behaviour hypotheses |
| `03_feature_engineering` | RFM derivation, log transforms, pipeline preview |
| `04_modeling_and_business_results` | Elbow curve, silhouette, cluster profiles, business translation |
| `05_deployment_and_consumption` | API demo, SQLite export, consuming new customer predictions |

---

## 9. API Deployment

The trained model is served via a FastAPI REST endpoint.

**Start the API:**
```bash
make api
# or: uvicorn src.insiders_loyalty_program.api:app --reload
```

**Health check:**
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

**Predict cluster for new customers:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "recency_days": 15,
        "frequency": 12,
        "monetary": 4500.0,
        "avg_ticket": 375.0,
        "total_items": 320
      }
    ]
  }'
```

Interactive API docs: `http://localhost:8000/docs`

---

## 10. Technologies

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-Learn 1.4 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn |
| Persistence | Joblib, SQLite |
| Config | TOML |
| Testing | Pytest |
| Containerisation | Docker, Docker Compose |
| Notebooks | JupyterLab |

---

## References

- Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025). O'Reilly Media.
- Dataset: [UCI Online Retail Dataset via Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- RFM methodology: Hughes, A.M. (1994). *Strategic Database Marketing*.
