"""Model training: rule-based baseline, k-search, stability, evaluation, save.

Clustering has no labels, so the supervised toolkit is adapted, not copied:

* **Baseline** is a rule-based RFM quantile scoring (the classic heuristic). It
  exists to justify whether K-Means earns its added complexity.
* **Tuning** is the search over ``k`` scored by three internal indices.
* **Cross-validation analogue** is a stability assessment: refit on bootstrap
  resamples and measure label agreement (Adjusted Rand Index).
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import Preprocessor

logger = config.get_logger(__name__)


class ModelTrainer:
    """Train, evaluate and serialize the RFM clustering pipeline."""

    def __init__(self) -> None:
        self.pre = Preprocessor()

    # ----------------------------- helpers -------------------------------- #
    def _transform(self, rfm: pd.DataFrame) -> np.ndarray:
        return self.pre.build_transformer().fit_transform(rfm[config.RFM_FEATURES])

    def _silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        if len(X) > config.SILHOUETTE_SAMPLE_SIZE:
            rng = np.random.default_rng(config.RANDOM_STATE)
            idx = rng.choice(len(X), size=config.SILHOUETTE_SAMPLE_SIZE, replace=False)
            return float(silhouette_score(X[idx], labels[idx]))
        return float(silhouette_score(X, labels))

    # ----------------------------- baseline ------------------------------- #
    def fit_baseline(self, rfm: pd.DataFrame, X: np.ndarray) -> dict:
        """Rule-based RFM quintile scoring; top quintile = heuristic Insiders."""
        rank = lambda s: s.rank(method="first")  # noqa: E731 - break qcut ties
        r = pd.qcut(rank(rfm["recency_days"]), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        f = pd.qcut(rank(rfm["frequency"]), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        m = pd.qcut(rank(rfm["monetary"]), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        score = (r + f + m).to_numpy()

        tiers = pd.qcut(pd.Series(score).rank(method="first"), 5, labels=False)
        is_top = score >= np.quantile(score, 0.8)
        revenue = rfm["monetary"].to_numpy()
        result = {
            "method": "RFM quintile scoring (R+F+M), top quintile = Insiders",
            "top_customers": int(is_top.sum()),
            "top_pct_customers": round(float(is_top.mean()) * 100, 2),
            "top_pct_revenue": round(float(revenue[is_top].sum() / revenue.sum()) * 100, 2),
            "silhouette": round(self._silhouette(X, tiers.to_numpy()), 4),
        }
        logger.info(
            "Baseline (RFM heuristic): top %.1f%% customers hold %.1f%% revenue (silhouette %.3f).",
            result["top_pct_customers"], result["top_pct_revenue"], result["silhouette"],
        )
        return result

    # --------------------------- model selection -------------------------- #
    def search_k(self, X: np.ndarray) -> tuple[list[dict], int]:
        """Grid-search ``k`` scored by silhouette, Davies-Bouldin, Calinski-Harabasz."""
        trials: list[dict] = []
        for k in config.CLUSTER_GRID:
            labels = KMeans(
                n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE
            ).fit_predict(X)
            trials.append(
                {
                    "k": k,
                    "silhouette": round(self._silhouette(X, labels), 4),
                    "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
                    "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
                }
            )
        # Select on silhouette; break ties by the lower (better) Davies-Bouldin.
        best = max(trials, key=lambda t: (t["silhouette"], -t["davies_bouldin"]))
        logger.info("Selected k=%d (silhouette %.3f).", best["k"], best["silhouette"])
        return trials, int(best["k"])

    def assess_stability(self, X: np.ndarray, k: int, reference: np.ndarray) -> dict:
        """Refit on bootstrap resamples; mean ARI vs the full-fit assignment."""
        rng = np.random.default_rng(config.RANDOM_STATE)
        size = int(len(X) * config.STABILITY_SAMPLE_FRAC)
        scores = []
        for _ in range(config.STABILITY_N_RESAMPLES):
            idx = rng.choice(len(X), size=size, replace=False)
            model = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=int(rng.integers(1e6)))
            model.fit(X[idx])
            scores.append(adjusted_rand_score(reference, model.predict(X)))
        mean_ari = round(float(np.mean(scores)), 4)
        logger.info("Stability: mean ARI %.3f over %d resamples.", mean_ari, config.STABILITY_N_RESAMPLES)
        return {
            "mean_adjusted_rand_index": mean_ari,
            "std": round(float(np.std(scores)), 4),
            "n_resamples": config.STABILITY_N_RESAMPLES,
            "sample_fraction": config.STABILITY_SAMPLE_FRAC,
        }

    # ------------------------------ fitting ------------------------------- #
    def fit(self, rfm: pd.DataFrame, k: int) -> tuple[Pipeline, np.ndarray]:
        """Fit the full pipeline and reorder centroids so 0 = highest monetary.

        Reordering the fitted ``cluster_centers_`` makes the serialized model
        emit value-ranked labels directly, so inference needs no external remap.
        """
        pipeline = Pipeline(
            steps=[
                ("preprocess", self.pre.build_transformer()),
                ("model", KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE)),
            ]
        )
        raw_labels = pipeline.fit_predict(rfm[config.RFM_FEATURES])
        km: KMeans = pipeline.named_steps["model"]

        mean_monetary = pd.Series(rfm["monetary"].to_numpy()).groupby(raw_labels).mean()
        order = mean_monetary.sort_values(ascending=False).index.to_numpy()
        km.cluster_centers_ = km.cluster_centers_[order]
        km.labels_ = np.argsort(order)[raw_labels]

        labels = pipeline.predict(rfm[config.RFM_FEATURES])
        assert np.array_equal(labels, km.labels_), "Centroid reorder broke label mapping."
        return pipeline, labels

    def name_segments(self, rfm: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
        """Rank 0 = Insiders; remaining ranks named by ascending mean recency."""
        df = rfm.assign(cluster=labels)
        recency = df.groupby("cluster")["recency_days"].mean().drop(index=0).sort_values()
        names = {0: config.INSIDER_SEGMENT}
        for i, cluster in enumerate(recency.index):
            names[int(cluster)] = (
                config.SECONDARY_SEGMENT_NAMES[i]
                if i < len(config.SECONDARY_SEGMENT_NAMES)
                else f"Tier {cluster}"
            )
        return names

    # ---------------------------- evaluation ------------------------------ #
    def evaluate(self, rfm: pd.DataFrame, labels: np.ndarray, X: np.ndarray) -> dict:
        """Internal indices, per-segment profile and boundary (error) analysis."""
        internal = {
            "silhouette": round(self._silhouette(X, labels), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
        }

        df = rfm.assign(cluster=labels)
        total_revenue = df["monetary"].sum()
        profile = (
            df.groupby("cluster")
            .agg(
                n_customers=("customer_id", "size"),
                recency_days=("recency_days", "mean"),
                frequency=("frequency", "mean"),
                monetary=("monetary", "mean"),
                avg_ticket=("avg_ticket", "mean"),
                total_items=("total_items", "mean"),
                revenue=("monetary", "sum"),
            )
            .round(2)
        )
        profile["pct_customers"] = (profile["n_customers"] / len(df) * 100).round(2)
        profile["pct_revenue"] = (profile["revenue"] / total_revenue * 100).round(2)

        # Error analysis: silhouette samples < 0 mark boundary/poorly-assigned points.
        sil = silhouette_samples(X, labels)
        boundary = pd.Series(sil < 0).groupby(labels).mean()
        profile["boundary_share"] = (boundary * 100).round(2)

        return {"internal_metrics": internal, "profile": profile, "overall_boundary_share": round(float((sil < 0).mean()) * 100, 2)}

    def to_business_metrics(self, profile: pd.DataFrame) -> dict:
        """Translate the top cluster into revenue-concentration business terms."""
        insiders = profile.loc[0]
        lift = insiders["pct_revenue"] / insiders["pct_customers"]
        return {
            "insiders_customers": int(insiders["n_customers"]),
            "insiders_pct_customers": float(insiders["pct_customers"]),
            "insiders_pct_revenue": float(insiders["pct_revenue"]),
            "insiders_revenue_lift": round(float(lift), 2),
            "insiders_total_revenue": float(round(insiders["revenue"], 2)),
            "insiders_avg_monetary": float(insiders["monetary"]),
        }

    # ------------------------------- save --------------------------------- #
    @staticmethod
    def _data_hash(rfm: pd.DataFrame) -> str:
        payload = pd.util.hash_pandas_object(rfm[config.RFM_FEATURES], index=False).values
        return hashlib.sha256(payload.tobytes()).hexdigest()[:16]

    def save(self, pipeline: Pipeline, card: dict) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.PIPELINE_PATH)
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
        logger.info("Saved pipeline -> %s and model card -> %s", config.PIPELINE_PATH, config.MODEL_CARD_PATH)

    # --------------------------- orchestration ---------------------------- #
    def run(self, rfm: pd.DataFrame) -> dict:
        """Baseline -> k-search -> fit -> stability -> evaluate -> persist."""
        X = self._transform(rfm)
        baseline = self.fit_baseline(rfm, X)
        trials, best_k = self.search_k(X)

        pipeline, labels = self.fit(rfm, best_k)
        stability = self.assess_stability(X, best_k, labels)
        names = self.name_segments(rfm, labels)
        evaluation = self.evaluate(rfm, labels, X)
        business = self.to_business_metrics(evaluation["profile"])

        scored = rfm.assign(cluster=labels)
        scored["segment"] = scored["cluster"].map(names)
        scored["is_insider"] = (scored["cluster"] == 0).astype(int)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(config.PROCESSED_FILE, index=False)
        logger.info("Wrote scored customer list -> %s", config.PROCESSED_FILE)

        profile = evaluation["profile"]
        segments = {
            int(c): {"name": names[int(c)], **{k: float(v) for k, v in profile.loc[c].items()}}
            for c in profile.index
        }
        card = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": {"kaggle": config.KAGGLE_DATASET, "raw_file": config.RAW_FILE.name},
            "data": {
                "n_customers": int(len(rfm)),
                "features": config.RFM_FEATURES,
                "data_hash": self._data_hash(rfm),
            },
            "preprocessing": "SimpleImputer(median) -> log1p -> StandardScaler",
            "model": {
                "algorithm": "KMeans",
                "selected_k": best_k,
                "n_init": config.KMEANS_N_INIT,
                "random_state": config.RANDOM_STATE,
            },
            "selection": {"grid": config.CLUSTER_GRID, "trials": trials},
            "stability": stability,
            "baseline": baseline,
            "evaluation": {
                "internal_metrics": evaluation["internal_metrics"],
                "overall_boundary_share_pct": evaluation["overall_boundary_share"],
            },
            "segments": segments,
            "segment_names": {str(r): n for r, n in names.items()},
            "business": business,
            "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        }
        self.save(pipeline, card)
        return card
