from __future__ import annotations

from fastapi import HTTPException


class GatewayError(HTTPException):
    code: str = "gateway_error"

    def __init__(self, status_code: int, detail: str, *, code: str | None = None) -> None:
        super().__init__(status_code=status_code, detail={"code": code or self.code, "message": detail})


class UnauthorizedError(GatewayError):
    code = "unauthorized"

    def __init__(self, detail: str = "invalid or missing API key") -> None:
        super().__init__(401, detail)


class RateLimitError(GatewayError):
    code = "rate_limited"

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(429, f"rate limit exceeded; retry after {retry_after_s:.1f}s")
        self.headers = {"Retry-After": str(max(1, int(retry_after_s)))}


class QueueFullError(GatewayError):
    code = "queue_full"

    def __init__(self, depth: int, max_depth: int) -> None:
        super().__init__(
            429,
            f"admission queue at capacity ({depth}/{max_depth}); try again shortly",
        )
        self.headers = {"Retry-After": "2"}


class QueueTimeoutError(GatewayError):
    code = "queue_timeout"

    def __init__(self) -> None:
        super().__init__(504, "request aged out of admission queue")


class EngineUnavailableError(GatewayError):
    code = "engine_unavailable"

    def __init__(self, detail: str = "inference engine unavailable") -> None:
        super().__init__(503, detail)


class CircuitOpenError(GatewayError):
    code = "circuit_open"

    def __init__(self) -> None:
        super().__init__(503, "engine circuit breaker open; shedding non-priority load")


class ValidationError(GatewayError):
    code = "invalid_request"

    def __init__(self, detail: str) -> None:
        super().__init__(400, detail)


class QuotaExceededError(GatewayError):
    code = "quota_exceeded"

    def __init__(self, api_key_id: str, limit: int) -> None:
        super().__init__(
            429,
            f"daily search quota of {limit} exceeded for key {api_key_id}",
        )
        self.headers = {"Retry-After": "3600"}
