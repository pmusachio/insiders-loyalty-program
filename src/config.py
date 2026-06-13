"""Central configuration: paths, dataset contract, hyperparameters and tuning grid.

Single source of truth for the pipeline. No values are hard-coded elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_FILE: Path = RAW_DIR / "Ecommerce.csv"
FEATURES_FILE: Path = PROCESSED_DIR / "rfm_features.parquet"
PROCESSED_FILE: Path = PROCESSED_DIR / "customer_segments.parquet"
# App-facing copy of the scored base: small (<150 KB), versioned, deployed to the
# Space so the app reads only from data/sample/ (never from raw/ or processed/).
SAMPLE_FILE: Path = SAMPLE_DIR / "customer_segments.parquet"
PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"

# --------------------------------------------------------------------------- #
# Data source (Kaggle)
# --------------------------------------------------------------------------- #
KAGGLE_DATASET: str = "carrie1/ecommerce-data"
# File name inside the Kaggle archive; renamed to RAW_FILE on download.
KAGGLE_ARCHIVE_CSV: str = "data.csv"

# Raw schema contract: canonical name -> source column.
RAW_COLUMNS: dict[str, str] = {
    "customer": "CustomerID",
    "invoice": "InvoiceNo",
    "date": "InvoiceDate",
    "quantity": "Quantity",
    "price": "UnitPrice",
}
RAW_DATE_FORMAT: str = "%d-%b-%y"

# --------------------------------------------------------------------------- #
# Feature contract
# --------------------------------------------------------------------------- #
# Engineered features (domain aggregation from raw transactions). The learned
# part of the representation (log1p + standardization params, cluster centroids)
# is fitted inside the serialized Pipeline, never computed by hand here.
RFM_FEATURES: list[str] = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_ticket",
    "total_items",
]
CUSTOMER_ID: str = "customer_id"

# Columns that must never reach the model matrix (identifiers / post-hoc labels).
# Enforced by tests as a guard against structural leakage.
NON_FEATURE_COLUMNS: list[str] = [CUSTOMER_ID, "cluster", "segment", "is_insider"]

# --------------------------------------------------------------------------- #
# Modeling hyperparameters and tuning grid
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
CLUSTER_GRID: list[int] = [3, 4, 5, 6, 7, 8, 9]
KMEANS_N_INIT: str | int = "auto"
SILHOUETTE_SAMPLE_SIZE: int = 10_000

# Stability assessment (clustering analogue of cross-validation): refit on
# bootstrap resamples and measure label agreement (Adjusted Rand Index).
STABILITY_N_RESAMPLES: int = 20
STABILITY_SAMPLE_FRAC: float = 0.8

# --------------------------------------------------------------------------- #
# Segment naming
# --------------------------------------------------------------------------- #
# Highest-monetary cluster is always the loyalty target.
INSIDER_SEGMENT: str = "Insiders"
# Remaining clusters are named by ascending mean recency (most active first).
SECONDARY_SEGMENT_NAMES: list[str] = [
    "Loyal Customers",
    "Promising",
    "At Risk",
    "Hibernating",
    "Lost",
    "Dormant",
]


def get_logger(name: str = "insiders") -> logging.Logger:
    """Return a module logger; configuration is owned by the CLI entrypoint."""
    return logging.getLogger(name)
