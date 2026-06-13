"""Smoke tests: I/O contract, shapes and absence of leaked columns."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src import config
from src.predict import Predictor, RFMInput, SegmentPrediction
from src.preprocessing import Preprocessor


def _raw_transactions() -> pd.DataFrame:
    """Two customers plus rows that the cleaning rules must drop."""
    return pd.DataFrame(
        [
            {"InvoiceNo": "100", "InvoiceDate": "01-Jan-16", "Quantity": 5, "UnitPrice": 10.0, "CustomerID": 1},
            {"InvoiceNo": "101", "InvoiceDate": "10-Jan-16", "Quantity": 3, "UnitPrice": 20.0, "CustomerID": 1},
            {"InvoiceNo": "102", "InvoiceDate": "05-Jan-16", "Quantity": 1, "UnitPrice": 50.0, "CustomerID": 2},
            {"InvoiceNo": "C103", "InvoiceDate": "06-Jan-16", "Quantity": 2, "UnitPrice": 30.0, "CustomerID": 1},  # cancelled
            {"InvoiceNo": "104", "InvoiceDate": "07-Jan-16", "Quantity": -2, "UnitPrice": 30.0, "CustomerID": 2},  # bad qty
            {"InvoiceNo": "105", "InvoiceDate": "08-Jan-16", "Quantity": 2, "UnitPrice": 30.0, "CustomerID": np.nan},  # no id
        ]
    )


def test_build_rfm_contract_and_cleaning() -> None:
    rfm = Preprocessor().build_rfm(_raw_transactions())
    assert list(rfm.columns) == [config.CUSTOMER_ID] + config.RFM_FEATURES
    assert rfm[config.CUSTOMER_ID].is_unique
    assert set(rfm[config.CUSTOMER_ID]) == {1, 2}
    # Cancelled invoice excluded -> customer 1 has 2 valid orders, not 3.
    assert int(rfm.loc[rfm[config.CUSTOMER_ID] == 1, "frequency"].iloc[0]) == 2


def test_transformer_consumes_only_rfm_features() -> None:
    """Guard against structural leakage: ids/labels never reach the model matrix."""
    transformer = Preprocessor().build_transformer()
    _, _, columns = transformer.transformers[0]
    assert columns == config.RFM_FEATURES
    assert not set(config.NON_FEATURE_COLUMNS).intersection(columns)


@pytest.fixture
def trained_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    high = pd.DataFrame(
        {
            "recency_days": rng.integers(1, 30, 30),
            "frequency": rng.integers(8, 20, 30),
            "monetary": rng.uniform(4000, 9000, 30),
            "avg_ticket": rng.uniform(30, 60, 30),
            "total_items": rng.integers(2000, 5000, 30),
        }
    )
    low = pd.DataFrame(
        {
            "recency_days": rng.integers(150, 300, 30),
            "frequency": rng.integers(1, 3, 30),
            "monetary": rng.uniform(50, 400, 30),
            "avg_ticket": rng.uniform(10, 25, 30),
            "total_items": rng.integers(20, 150, 30),
        }
    )
    rfm = pd.concat([high, low], ignore_index=True)
    pipeline = Pipeline(
        [("preprocess", Preprocessor().build_transformer()), ("model", KMeans(n_clusters=2, n_init="auto", random_state=0))]
    )
    pipeline.fit(rfm[config.RFM_FEATURES])
    pipeline_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    card_path = tmp_path / "model_card.json"
    card_path.write_text(json.dumps({"segment_names": {"0": "Insiders", "1": "At Risk"}}))
    return pipeline_path, card_path


def test_predictor_output_contract(trained_artifacts) -> None:
    predictor = Predictor(*trained_artifacts)
    result = predictor.predict(RFMInput(recency_days=10, frequency=12, monetary=5000, avg_ticket=40, total_items=2500))
    assert isinstance(result, SegmentPrediction)
    assert isinstance(result.cluster_id, int)
    assert result.segment in {"Insiders", "At Risk"}
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.is_insider, bool)
    assert len(result.feature_contributions) == len(config.RFM_FEATURES)
    assert {fc.feature for fc in result.feature_contributions} == set(config.RFM_FEATURES)


def test_predictor_accepts_mapping(trained_artifacts) -> None:
    predictor = Predictor(*trained_artifacts)
    payload = {f: 1.0 for f in config.RFM_FEATURES}
    assert isinstance(predictor.predict(payload), SegmentPrediction)


def test_predictor_rejects_invalid_input(trained_artifacts) -> None:
    predictor = Predictor(*trained_artifacts)
    with pytest.raises(ValueError):
        predictor.predict({"recency_days": 1})  # missing features
    with pytest.raises(ValueError):
        predictor.predict(RFMInput(recency_days=-1, frequency=1, monetary=1, avg_ticket=1, total_items=1))


@pytest.mark.skipif(not config.PIPELINE_PATH.exists(), reason="run `python -m src.pipeline` first")
def test_trained_pipeline_flags_insider_profile() -> None:
    result = Predictor().predict(
        RFMInput(recency_days=17, frequency=14, monetary=8000, avg_ticket=38, total_items=4600)
    )
    assert result.is_insider and result.segment == "Insiders"
