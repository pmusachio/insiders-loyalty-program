# Notebooks

Run the notebooks in order for the full end-to-end story:

```bash
cd ..   # project root
jupyter lab
```

| # | Notebook | What it covers |
|---|---|---|
| 00 | `00_business_understanding.ipynb` | Problem framing, ML type, success metrics |
| 01 | `01_data_understanding.ipynb` | Schema, missing values, distributions, data quality |
| 02 | `02_exploratory_analysis.ipynb` | Pareto analysis, RFM scatter plots, hypothesis testing |
| 03 | `03_feature_engineering.ipynb` | RFM derivation, log transform, sklearn Pipeline preview |
| 04 | `04_modeling_and_business_results.ipynb` | Elbow curve, silhouette, cluster profiles, business translation |
| 05 | `05_deployment_and_consumption.ipynb` | API test, SQLite export, production architecture |

All notebooks import from `src/insiders_loyalty_program/`, keeping the visual narrative aligned with the reproducible pipeline.

Figures saved by notebooks are written to `../reports/figures/`.
