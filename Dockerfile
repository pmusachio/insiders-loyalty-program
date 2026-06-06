FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/

# Create output directories
RUN mkdir -p models reports/figures

ENV PYTHONPATH=src

EXPOSE 8000

CMD ["uvicorn", "insiders_loyalty_program.api:app", "--host", "0.0.0.0", "--port", "8000"]
