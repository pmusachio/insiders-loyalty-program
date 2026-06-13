# Notebooks — Development Record

These notebooks are the original exploration record, kept for transparency on how
the solution was reached: business framing → data understanding → EDA → feature
engineering → modeling and results.

They are **not** part of the production pipeline. The reproducible logic was
consolidated into `src/` (run with `python -m src.pipeline`); notebook imports
reference the earlier module layout and are preserved unchanged as a historical
artifact.

| # | Notebook | Focus |
|---|---|---|
| 00 | `00_business_understanding` | Problem framing, ML-vs-heuristic, success criteria |
| 01 | `01_data_understanding` | Schema, missing values, distributions |
| 02 | `02_exploratory_analysis` | Pareto/revenue concentration, RFM behaviour |
| 03 | `03_feature_engineering` | RFM derivation, skew/log transform |
| 04 | `04_modeling_and_business_results` | k-search, silhouette, cluster profiles |
| 05 | `05_deployment_and_consumption` | Serving and consumption walkthrough |
