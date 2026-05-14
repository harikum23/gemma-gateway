from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from gateway.auth import ApiKeyStore, ensure_bootstrap_key
from gateway.circuit import CircuitRegistry
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


def _json_sink(message) -> None:
    """loguru sink emitting one JSON object per line for log aggregators."""
    record = message.record
    payload = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"] is not None:
        payload["exception"] = str(record["exception"])
    sys.stderr.write(json.dumps(payload) + "\n")
    sys.stderr.flush()


def _configure_logging(level: str, fmt: str = "text") -> None:
    logger.remove()
    if fmt == "json":
        logger.add(_json_sink, level=level.upper(), enqueue=False, backtrace=False, diagnose=False)
    else:
        logger.add(
            sys.stderr,
            level=level.upper(),
            enqueue=False,
            backtrace=False,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<5} | {name}:{function}:{line} | {message}",
        )


async def _warmup(engine, default_model: str, timeout_s: float = 10.0) -> None:
    """Warm up the engine without blocking startup if it's slow/unavailable."""
    try:
        await asyncio.wait_for(
            engine.generate(
                model=default_model,
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_tokens=1,
                stop=None,
                tools=None,
                response_format="text",
                json_schema=None,
            ),
            timeout=timeout_s,
        )
        logger.info("model warmup complete")
    except asyncio.TimeoutError:
        logger.warning("model warmup timed out after {}s (gateway still serving)", timeout_s)
    except Exception as exc:
        logger.warning("model warmup failed (gateway still serving): {}", exc)


async def _metrics_prune_loop(metrics: MetricsStore, retention_days: int, interval_s: int) -> None:
    """Periodically prune old metrics so the DB stays bounded."""
    while True:
        try:
            await asyncio.sleep(interval_s)
            pruned = metrics.prune(retention_days=retention_days)
            if pruned:
                logger.info("metrics pruned {} rows older than {} days", pruned, retention_days)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("metrics prune failed: {}", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level, settings.log_format)
    logger.info("gemma-gateway starting (engine={}, model={})", settings.engine_type, settings.default_model)

    api_keys = ApiKeyStore(settings.api_key_db_path)
    ensure_bootstrap_key(
        api_keys,
        settings.bootstrap_api_key,
        log_generated_key=settings.log_bootstrap_key,
    )

    engine = build_engine(settings)
    queue = AdmissionQueue(
        concurrency=settings.admission_concurrency,
        max_depth=settings.admission_max_depth,
        default_max_wait_ms=settings.admission_max_wait_ms,
    )
    queue.start()

    circuit = CircuitRegistry(
        error_threshold=settings.circuit_error_threshold,
        window_s=settings.circuit_window_s,
        reset_s=settings.circuit_reset_s,
    )
    rate_limiter = TokenBucketRateLimiter(rps=settings.default_rps_limit)
    metrics = MetricsStore(settings.data_dir / "metrics.db")

    # Redis client — optional; search caching/quota degrades gracefully if absent.
    redis_client = None
    redis_required = bool(settings.redis_required and settings.redis_url)
    if _REDIS_AVAILABLE and settings.redis_url:
        try:
            redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
            await redis_client.ping()
            logger.info("redis connected: {}", settings.redis_url)
        except Exception as exc:
            if redis_required:
                logger.error("redis required but unreachable: {}", exc)
                raise
            logger.warning("redis unavailable (search caching/quotas best-effort): {}", exc)
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
    app.state.redis_required = redis_required
    app.state.trace_db = trace_db

    # Prune old metrics on startup, then run a periodic loop in the background.
    pruned = metrics.prune(retention_days=settings.metrics_retention_days)
    if pruned:
        logger.info("metrics pruned {} rows older than {} days", pruned, settings.metrics_retention_days)
    prune_task = asyncio.create_task(
        _metrics_prune_loop(
            metrics,
            settings.metrics_retention_days,
            settings.metrics_prune_interval_s,
        )
    )

    # Run warmup in background — don't block readiness on slow models.
    warmup_task = asyncio.create_task(_warmup(engine, settings.default_model))

    try:
        yield
    finally:
        logger.info("gemma-gateway shutting down")
        for t in (prune_task, warmup_task):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await queue.stop()
        await engine.aclose()
        from gateway.tools.http_client import reset_pool
        await reset_pool()
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

    @app.middleware("http")
    async def _limit_body_size(request: Request, call_next):
        """Reject oversized request bodies before they hit Pydantic.

        Trusts Content-Length when present (cheap pre-read check). Streaming
        clients without Content-Length are passed through and bounded only
        by the underlying ASGI server's defaults.
        """
        settings = getattr(request.app.state, "settings", None)
        limit = getattr(settings, "max_request_bytes", 1_048_576) if settings else 1_048_576
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > limit:
                    return JSONResponse(
                        status_code=413,
                        content={"error": f"request body exceeds {limit} bytes"},
                    )
            except ValueError:
                pass
        return await call_next(request)

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(embed.router)
    app.include_router(admin.router)
    app.include_router(agent.router)
    app.include_router(portal.router)
    app.include_router(chat.router)
    return app


app = create_app()
