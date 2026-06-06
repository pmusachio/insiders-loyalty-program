# Deployment Guide

The project delivers an actionable customer list for the Insiders loyalty programme and a lightweight layer for BI consumption.

## Delivery Channels

| Channel | Output | Consumer |
|---|---|---|
| Batch job | `reports/cluster_assignments.csv` | CRM import, data team |
| SQLite export | `reports/insiders_segments.sqlite` | Metabase, Power BI, Tableau |
| REST API | `http://localhost:8000/predict` | CRM real-time scoring |
| Docker | `docker compose up api` | Any deployment environment |

## Local Execution

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-api.txt

# Run training pipeline
make train

# Export to SQLite
make export

# Start API
make api
```

## Docker Execution

```bash
# Build image and run training
docker compose run --rm train

# Start API service
docker compose up api
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict cluster for RFM features |

## BI Connection

Connect Metabase, Power BI, or Tableau directly to `reports/insiders_segments.sqlite`.

The `customer_segments` table contains one row per customer with columns:
- `customer_id` — customer identifier
- `recency_days`, `frequency`, `monetary`, `avg_ticket`, `total_items` — RFM features
- `cluster` — numeric cluster label (0 = Insiders by default)
- `segment` — human-readable label (INSIDERS, Loyalists, At-Risk, Churned)
- `is_insider` — binary flag (1 = Insider, 0 = other)

## Cloud Architecture (Production)

```
S3 (raw data) → EC2 / GitHub Actions (monthly training) → RDS/Postgres
                                                          ↓
                                                    Metabase Dashboard
                                                    FastAPI + Docker (real-time)
```
