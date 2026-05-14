from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from gateway.auth import ApiKeyStore, ensure_bootstrap_key
from gateway.circuit import CircuitBreaker
from gateway.engines.factory import build_engine
from gateway.queue import AdmissionQueue
from gateway.rate_limit import TokenBucketRateLimiter
from gateway.agent import trace as agent_trace
from gateway.routers import admin, agent, chat, embed, generate, health, portal
from gateway.settings import get_settings
from gateway.telemetry import MetricsStore

try:
    import redis.asyncio as aioredis  # type: ignore[import]
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        enqueue=False,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<5} | {name}:{function}:{line} | {message}",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info("gemma-gateway starting (engine={}, model={})", settings.engine_type, settings.default_model)

    api_keys = ApiKeyStore(settings.api_key_db_path)
    ensure_bootstrap_key(api_keys, settings.bootstrap_api_key)

    engine = build_engine(settings)
    queue = AdmissionQueue(
        concurrency=settings.admission_concurrency,
        max_depth=settings.admission_max_depth,
        default_max_wait_ms=settings.admission_max_wait_ms,
    )
    queue.start()

    circuit = CircuitBreaker(
        error_threshold=settings.circuit_error_threshold,
        window_s=settings.circuit_window_s,
        reset_s=settings.circuit_reset_s,
    )
    rate_limiter = TokenBucketRateLimiter(rps=settings.default_rps_limit)
    metrics = MetricsStore(settings.data_dir / "metrics.db")

    # Redis client — optional; search caching/quota degrades gracefully if absent
    redis_client = None
    if _REDIS_AVAILABLE and settings.redis_url:
        try:
            redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("redis connected: {}", settings.redis_url)
        except Exception as exc:
            logger.warning("redis unavailable (search caching disabled): {}", exc)
            redis_client = None

    # DuckDB trace store for agent runtime
    import duckdb  # type: ignore[import]
    trace_db = duckdb.connect(str(settings.data_dir / "traces.db"))
    agent_trace.ensure_table(trace_db)

    app.state.settings = settings
    app.state.api_keys = api_keys
    app.state.engine = engine
    app.state.queue = queue
    app.state.circuit = circuit
    app.state.rate_limiter = rate_limiter
    app.state.metrics = metrics
    app.state.redis = redis_client
    app.state.trace_db = trace_db

    try:
        await engine.generate(
            model=settings.default_model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=1,
            stop=None,
            tools=None,
            response_format="text",
            json_schema=None,
        )
        logger.info("model warmup complete")
    except Exception as exc:
        logger.warning("model warmup failed (gateway still starting): {}", exc)

    # Prune old metrics on startup so the DB doesn't grow unbounded.
    pruned = metrics.prune(retention_days=settings.metrics_retention_days)
    if pruned:
        logger.info("metrics pruned {} rows older than {} days", pruned, settings.metrics_retention_days)

    try:
        yield
    finally:
        logger.info("gemma-gateway shutting down")
        await queue.stop()
        await engine.aclose()
        if redis_client is not None:
            await redis_client.aclose()
        try:
            trace_db.close()
        except Exception:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="gemma-gateway",
        version="0.1.0",
        description="High-traffic local LLM service — generate, embed, rerank.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(embed.router)
    app.include_router(admin.router)
    app.include_router(agent.router)
    app.include_router(portal.router)
    app.include_router(chat.router)
    return app


app = create_app()
