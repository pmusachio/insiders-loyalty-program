"""Inference contract: load the serialized pipeline and score a customer.

The serialized artifact bundles preprocessing and K-Means, so training and
serving share one transformation (no training-serving skew).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Union

import joblib
import numpy as np
import pandas as pd

from . import config


@dataclass
class RFMInput:
    """Explicit input schema for a single customer."""

    recency_days: float
    frequency: float
    monetary: float
    avg_ticket: float
    total_items: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])[config.RFM_FEATURES]


@dataclass
class FeatureContribution:
    """How a feature defines the assigned segment, and where the input sits.

    Both values are standardized (z-score in the model's log space): the segment
    value is the centroid coordinate; the input value is the customer's position.
    """

    feature: str
    segment_zscore: float
    input_zscore: float


@dataclass
class SegmentPrediction:
    """Explicit output schema returned by :meth:`Predictor.predict`."""

    cluster_id: int
    segment: str
    is_insider: bool
    confidence: float
    feature_contributions: list[FeatureContribution]


Features = Union[RFMInput, Mapping[str, float], pd.DataFrame]


class Predictor:
    """Load the trained pipeline and expose a typed prediction contract."""

    def __init__(
        self,
        pipeline_path: Path = config.PIPELINE_PATH,
        card_path: Path = config.MODEL_CARD_PATH,
    ) -> None:
        if not Path(pipeline_path).exists():
            raise FileNotFoundError(
                f"Trained pipeline not found at {pipeline_path}. Run `python -m src.pipeline` first."
            )
        self.pipeline = joblib.load(pipeline_path)
        self.preprocess = self.pipeline.named_steps["preprocess"]
        self.kmeans = self.pipeline.named_steps["model"]
        card = json.loads(Path(card_path).read_text(encoding="utf-8")) if Path(card_path).exists() else {}
        self.segment_names = {int(k): v for k, v in card.get("segment_names", {}).items()}

    def _coerce(self, features: Features) -> pd.DataFrame:
        if isinstance(features, RFMInput):
            frame = features.to_frame()
        elif isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            frame = pd.DataFrame([dict(features)])

        missing = set(config.RFM_FEATURES).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing RFM features: {sorted(missing)}")
        frame = frame[config.RFM_FEATURES].apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            raise ValueError("All RFM features must be numeric.")
        if (frame < 0).any().any():
            raise ValueError("RFM features must be non-negative.")
        return frame

    def predict(self, features: Features) -> SegmentPrediction:
        """Score one customer into a value-ranked segment with interpretation."""
        frame = self._coerce(features)
        cluster_id = int(self.pipeline.predict(frame)[0])

        x = np.asarray(self.preprocess.transform(frame))[0]
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - x, axis=1)
        weights = np.exp(-distances)
        confidence = float(weights[cluster_id] / weights.sum())

        centroid = self.kmeans.cluster_centers_[cluster_id]
        contributions = sorted(
            (
                FeatureContribution(feature, round(float(centroid[i]), 3), round(float(x[i]), 3))
                for i, feature in enumerate(config.RFM_FEATURES)
            ),
            key=lambda c: abs(c.segment_zscore),
            reverse=True,
        )
        return SegmentPrediction(
            cluster_id=cluster_id,
            segment=self.segment_names.get(cluster_id, f"Cluster {cluster_id}"),
            is_insider=cluster_id == 0,
            confidence=round(confidence, 4),
            feature_contributions=contributions,
        )
