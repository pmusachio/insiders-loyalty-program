"""Single-command pipeline: download -> preprocess -> baseline -> fit -> evaluate -> save.

Idempotent: runs from an empty ``data/`` (downloading raw on demand) or reuses an
existing raw file. Entrypoint: ``python -m src.pipeline``.
"""

from __future__ import annotations

import argparse
import logging

from . import config
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .train import ModelTrainer

logger = config.get_logger("insiders.pipeline")


def run() -> dict:
    raw = DataLoader().load()
    rfm = Preprocessor().run(raw)
    return ModelTrainer().run(rfm)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    argparse.ArgumentParser(description="Run the RFM clustering pipeline end to end.").parse_args(argv)

    card = run()
    business = card["business"]
    logger.info(
        "Pipeline complete | k=%d | Insiders %.1f%% of customers hold %.1f%% of revenue (%.1fx lift).",
        card["model"]["selected_k"],
        business["insiders_pct_customers"],
        business["insiders_pct_revenue"],
        business["insiders_revenue_lift"],
    )


if __name__ == "__main__":
    main()
