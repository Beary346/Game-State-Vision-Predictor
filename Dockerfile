# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Stable environment inside the container: unbuffered logs for `docker compose
# logs`, no stray .pyc files, and PYTHONPATH=/app so `python -m src.pipeline.*`
# and `uvicorn app.labeler:app` resolve from anywhere.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install CPU-only PyTorch BEFORE the project requirements: the default PyPI
# torch wheel bundles CUDA (~2.6 GB) which Gold training and inference never
# need. Installing torch first satisfies the torch>=2.1.0 pin in
# requirements/ml.txt, so pip never pulls the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0"

# Then the declared project stack (ml + viz + workflow requirements; base.txt
# also carries fastapi/uvicorn for the label-review app).
COPY requirements.txt requirements/ ./
RUN pip install -r requirements.txt

# Application code. data/ is intentionally NOT copied: it is bind-mounted from
# the host (docker-compose) so label scaffolds and trained artifacts survive
# container rebuilds and are shared between the app and training job.
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY pyproject.toml ./

# Run as a non-root user; the host data/model/outputs mounts are
# world-writable in this project's workspace, so uid 1000 can read and write
# the shared label/artifact volumes.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /mlruns /mlartifacts \
    && chown -R appuser:appuser /app /mlruns /mlartifacts
USER appuser

# Default entrypoint: the label-review app. docker-compose overrides this
# per service (mlflow server, gold training job).
EXPOSE 8765
CMD ["uvicorn", "app.labeler:app", "--host", "0.0.0.0", "--port", "8765"]