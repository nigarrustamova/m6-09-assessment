# ─────────────────────────────────────────────────────────────────────────────
# Cat Detector — production inference image
#
# Build:
#   docker build -t nigarrustamova/cat-detector:final -f container/Dockerfile .
#
# Run (info):
#   docker run --rm nigarrustamova/cat-detector:final info
#
# Run (predict):
#   docker run --rm \
#     -v /absolute/path/to/holdout:/data/input:ro \
#     -v /absolute/path/to/results:/data/output \
#     nigarrustamova/cat-detector:final predict
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory — all app files live here inside the container
WORKDIR /app

# ── 1. Install Python dependencies ────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of
# code changes — rebuilds after a code edit skip the pip install step.
COPY container/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── 2. Copy application code ──────────────────────────────────────────────────
COPY container/app /app/app

# ── 3. Copy ONNX model weights ────────────────────────────────────────────────
COPY container/models /app/models

# ── 4. Copy student identification record ─────────────────────────────────────
COPY container/STUDENT.json /app/STUDENT.json

# ── 5. Entrypoint ─────────────────────────────────────────────────────────────
# The subcommand (info / predict) is passed as CMD at docker run time.
ENTRYPOINT ["python", "/app/app/cli.py"]
