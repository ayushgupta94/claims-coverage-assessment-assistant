# Single-stage build is intentional here: this is a Python service with no
# compiled build step (no frontend bundling, no Go/Rust binary), so a
# multi-stage build would add complexity without a real image-size benefit.
FROM python:3.12-slim

WORKDIR /app

# System deps for scikit-learn/numpy wheels build correctly on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Azure Container Apps (and most orchestrators) send SIGTERM on shutdown;
# uvicorn handles that gracefully by default, no extra signal handling needed.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
