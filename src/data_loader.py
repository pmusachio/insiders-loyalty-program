"""Raw data acquisition: Kaggle download and reproducible CSV read."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

logger = config.get_logger(__name__)


class DataLoader:
    """Fetch the raw transactional dataset and load it into memory.

    The Kaggle download is triggered only when ``raw/`` is empty, so the
    pipeline is idempotent and offline-friendly once the file exists.
    """

    def __init__(self, raw_file: Path = config.RAW_FILE) -> None:
        self.raw_file = raw_file

    def download(self) -> Path:
        """Download the dataset from Kaggle into ``data/raw/``.

        Requires ``~/.kaggle/kaggle.json``. Imported lazily so the rest of the
        pipeline runs without the Kaggle client installed.
        """
        if self.raw_file.exists():
            logger.info("Raw file already present at %s; skipping download.", self.raw_file)
            return self.raw_file

        from kaggle.api.kaggle_api_extended import KaggleApi

        self.raw_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from Kaggle...", config.KAGGLE_DATASET)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            config.KAGGLE_DATASET, path=str(self.raw_file.parent), unzip=True
        )

        archive_csv = self.raw_file.parent / config.KAGGLE_ARCHIVE_CSV
        if archive_csv.exists() and archive_csv != self.raw_file:
            archive_csv.rename(self.raw_file)
        if not self.raw_file.exists():
            raise FileNotFoundError(
                f"Download finished but {self.raw_file.name} was not produced."
            )
        logger.info("Raw dataset ready at %s", self.raw_file)
        return self.raw_file

    def load(self) -> pd.DataFrame:
        """Read the raw CSV, downloading first if necessary."""
        if not self.raw_file.exists():
            self.download()

        for encoding in ("utf-8", "latin1"):
            try:
                df = pd.read_csv(self.raw_file, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - both encodings failed
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode raw CSV.")

        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        logger.info("Loaded %d raw transaction rows.", len(df))
        return df
