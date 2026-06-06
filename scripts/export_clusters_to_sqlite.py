"""Export customer clusters to a SQLite database for BI tools."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cluster assignments to SQLite.")
    parser.add_argument("--input", default="reports/cluster_assignments.csv")
    parser.add_argument("--output", default="reports/insiders_segments.sqlite")
    parser.add_argument("--table", default="customer_segments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Run the training pipeline first. Missing: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    with sqlite3.connect(output_path) as connection:
        df.to_sql(args.table, connection, if_exists="replace", index=False)

    print(f"Exported {len(df)} rows to {output_path}::{args.table}")


if __name__ == "__main__":
    main()
