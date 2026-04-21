FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip build && \
    pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels gemma-gateway \
    && rm -rf /wheels

RUN useradd --system --create-home --uid 10001 gateway \
    && mkdir -p /data \
    && chown -R gateway:gateway /data
USER gateway
WORKDIR /home/gateway

ENV GATEWAY_PORT=8080 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_DATA_DIR=/data

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/v1/health || exit 1

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
