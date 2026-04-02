FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --no-dev

EXPOSE 8000 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["uv", "run", "python", "-c", "import os; import httpx; r = httpx.get(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health'); r.raise_for_status()"]

CMD ["uv", "run", "python", "main.py"]
