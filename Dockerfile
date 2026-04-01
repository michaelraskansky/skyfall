FROM python:3.12-slim

LABEL maintainer="debris-tracker"
LABEL description="Real-Time Aerospace Debris & Industrial Anomaly Tracker"

WORKDIR /app

# Install OS-level dependencies (none needed for now, but layer is cached).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

# Expose the FastAPI emergency webhook port.
EXPOSE 8000

# Health-check against the FastAPI liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the orchestrator.
CMD ["python", "main.py"]
