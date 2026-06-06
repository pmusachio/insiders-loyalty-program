# Raw Data

> **The CSV file is NOT committed to this repository** (40 MB, exceeds GitHub's 25 MB limit).  
> Download it before running the pipeline.

**Source:** [E-Commerce Data — Kaggle (UCI ML Repository)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

**Expected file:** `data/raw/Ecommerce.csv`

---

## Option A — Kaggle API (recommended)

```bash
pip install kaggle

# Generate your token at: Kaggle → Account → API → Create New Token
# Place kaggle.json at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p data/raw
kaggle datasets download -d carrie1/ecommerce-data --unzip -p data/raw
mv data/raw/data.csv data/raw/Ecommerce.csv 2>/dev/null || true
```

## Option B — Manual download

1. Visit https://www.kaggle.com/datasets/carrie1/ecommerce-data
2. Click **Download**
3. Unzip the file
4. Rename `data.csv` to `Ecommerce.csv`
5. Place it at `data/raw/Ecommerce.csv`

## Option C — Google Colab

```python
from google.colab import files
files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data/raw
!kaggle datasets download -d carrie1/ecommerce-data --unzip -p data/raw/
!mv data/raw/data.csv data/raw/Ecommerce.csv 2>/dev/null || true
```
