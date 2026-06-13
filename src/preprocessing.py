"""RFM feature engineering and the reproducible transformation pipeline.

Feature provenance
------------------
* **Engineered (domain rule):** ``recency_days``, ``frequency``, ``monetary``,
  ``avg_ticket``, ``total_items`` are aggregated from raw transactions in
  :meth:`Preprocessor.build_rfm`.
* **Learned (fitted in-pipeline):** the ``log1p`` + standardization parameters
  returned by :meth:`Preprocessor.build_transformer` are fitted as part of the
  serialized sklearn ``Pipeline`` — never as loose mutations on a DataFrame.

Data-leakage assessment
-----------------------
This is unsupervised: there is no target, so target leakage is structurally
impossible. The only temporal dependency is ``recency_days``, measured against a
snapshot reference date (max invoice date + 1 day). In production, RFM must be
recomputed as-of the scoring date; the app accepts RFM values directly, so no
look-ahead is introduced. ``customer_id`` is an identifier and is excluded from
the model matrix (enforced in tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config

logger = config.get_logger(__name__)


class Preprocessor:
    """Turn raw transactions into a customer-level RFM table and expose the
    sklearn transformer that standardizes those features for clustering."""

    def __init__(self) -> None:
        self.cols = config.RAW_COLUMNS

    def build_rfm(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw transactions into one RFM row per customer.

        Cleaning rules: drop missing customers, parse dates, keep positive
        quantity and price, and remove cancelled invoices (``InvoiceNo``
        prefixed with ``C``) before aggregating.
        """
        customer, invoice = self.cols["customer"], self.cols["invoice"]
        date, quantity, price = self.cols["date"], self.cols["quantity"], self.cols["price"]

        missing = {customer, invoice, date, quantity, price}.difference(raw.columns)
        if missing:
            raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

        df = raw.dropna(subset=[customer]).copy()
        df[date] = pd.to_datetime(df[date], format=config.RAW_DATE_FORMAT, errors="coerce")
        df = df[df[date].notna()]
        df = df[(df[quantity] > 0) & (df[price] > 0)]
        df = df[~df[invoice].astype(str).str.startswith("C")]
        df["gross_revenue"] = df[quantity] * df[price]

        reference_date = df[date].max() + pd.Timedelta(days=1)
        rfm = (
            df.groupby(customer)
            .agg(
                recency_days=(date, lambda s: (reference_date - s.max()).days),
                frequency=(invoice, "nunique"),
                monetary=("gross_revenue", "sum"),
                avg_ticket=("gross_revenue", "mean"),
                total_items=(quantity, "sum"),
            )
            .reset_index()
            .rename(columns={customer: config.CUSTOMER_ID})
        )
        rfm[config.CUSTOMER_ID] = rfm[config.CUSTOMER_ID].astype("int64")
        logger.info("Built RFM table for %d customers.", len(rfm))
        return rfm

    @staticmethod
    def build_transformer() -> ColumnTransformer:
        """ColumnTransformer: median-impute -> log1p -> standardize.

        RFM features are right-skewed; ``log1p`` compresses the tail before
        distance-based clustering. All parameters are fitted, never hand-set.
        """
        numeric = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[("rfm", numeric, config.RFM_FEATURES)],
            remainder="drop",
        )

    def write_sample(self, rfm: pd.DataFrame) -> None:
        """Persist a small reference sample for the app (slider ranges, examples)."""
        n = min(config.SAMPLE_ROWS, len(rfm))
        sample = rfm.sample(n=n, random_state=config.RANDOM_STATE)
        config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(config.SAMPLE_FILE, index=False)
        logger.info("Wrote reference sample (%d rows) to %s", n, config.SAMPLE_FILE)

    def run(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Build RFM, persist the processed feature table and the app sample."""
        rfm = self.build_rfm(raw)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        rfm.to_parquet(config.FEATURES_FILE, index=False)
        logger.info("Wrote processed RFM features to %s", config.FEATURES_FILE)
        self.write_sample(rfm)
        return rfm
