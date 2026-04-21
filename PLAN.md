# gemma-gateway — High-Traffic Local LLM Service

## Purpose
A single, shareable HTTP service that exposes Gemma (and future models) as LLM primitives to every project on this machine — SMA, Indian-stock-analyzer, AgentFlow, and future apps. Owns model runtime, batching, queuing, auth, and observability. Stateless with respect to business data; does **not** store prompts, embeddings, or domain content.

## Non-goals
- No RAG logic, no evidence-pack assembly, no trading rules. Those live in consumer apps.
- No model fine-tuning in v1 (add a separate training plan later).
- Not a general-purpose OpenAI proxy; speaks a minimal, explicit contract.

---

## Top-level design decisions

| Decision | Choice | Reason |
|---|---|---|
| Model default (Mac mini M4 deploy) | **Gemma 4 E4B-it** via MLX-LM; tier-up to `gemma4-26b-it` on 32GB, `gemma4-31b-it` on M4 Pro 48GB+ | Fits base M4 mini without starving OS/Postgres/app workloads; swap via env var as hardware grows |
| Model default (CUDA host) | `gemma4-31b-it` (bf16 or AWQ) via vLLM | Used when gateway moves off the Mac mini to a dedicated GPU box |
| Inference engine (prod) | **vLLM** (CUDA hosts) or **MLX-LM** (Apple Silicon) | Both provide continuous batching + tensor parallel |
| Inference engine (dev) | **Ollama** | Fast local iteration, model pull UX |
| API framework | **FastAPI** (Python 3.12) + **uvicorn** with multiple workers | Async, typed, proven |
| Embedding model | **nomic-embed-text-v1.5** or **bge-large-en-v1.5** | Strong English retrieval quality, small VRAM |
| Rerank model (optional) | **bge-reranker-v2-m3** | Cross-encoder, small, high precision |
| Reverse proxy | **Caddy** (auto-TLS) or **Traefik** | Simple ops |
| Queue / backpressure | **in-process asyncio.Queue** + **Redis** for cross-replica | Redis required only when scaling out |
| Auth | **API keys** (HMAC-verified), optional mTLS | Simple, rotatable |
| Observability | **Prometheus** + **Grafana** + **OpenTelemetry** traces | Standard stack |
| Config | **pydantic-settings** + `.env` + Keychain for secrets | Type-safe, no plaintext secrets |
| Packaging | **Docker Compose** primary, **systemd** for bare-metal, **k8s** manifests optional | Portable |

---

## Architecture

### Mac mini M4 deployment (hybrid — inference native, services in Docker)

```
┌─────────────────────────── macOS host ───────────────────────────┐
│                                                                  │
│  Native processes (Metal-accelerated, not in Docker):            │
│  ┌──────────────────────────────────────────┐                    │
│  │ MLX-LM server   :11434                   │  gemma4-e4b-it     │
│  │ (or native Ollama)                       │                    │
│  └──────────────────────────────────────────┘                    │
│                        ▲                                         │
│                        │ host.docker.internal:11434              │
│                        │                                         │
│  ┌─────────────────────┴────────────────────────┐                │
│  │  Docker Desktop (or OrbStack) VM             │                │
│  │                                              │                │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │                │
│  │  │ Caddy    │─►│ gateway  │─►│ Redis    │    │                │
│  │  │ :443     │  │ FastAPI  │  │          │    │                │
│  │  └──────────┘  │ :8080    │  └──────────┘    │                │
│  │                └──────────┘                  │                │
│  │                                              │                │
│  │  ┌──────────┐  ┌──────────┐                  │                │
│  │  │Prometheus│  │ Grafana  │                  │                │
│  │  └──────────┘  └──────────┘                  │                │
│  └──────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────┘
                         ▲
                         │ HTTPS + API key
                         │
          Client apps (SMA, ISA, AgentFlow)
```

### Why hybrid, not all-in-Docker (critical)
Docker Desktop on macOS runs containers inside a Linux VM. That VM **cannot access Apple's Metal GPU or Neural Engine**. Running `mlx-lm` or `ollama` inside a Docker container on a Mac forces CPU-only inference, collapsing throughput by 5–20×.

Concrete numbers on Mac mini M4 base:
| Placement | gemma4-e4b-it tok/s | gemma4-26b-it tok/s |
|---|---|---|
| Native (Metal via MLX) | 60–120 | 10–18 |
| In Docker (CPU-only) | 8–15 | 0.5–2 |

Hybrid keeps the inference engine native (full Metal) while containerizing everything else. Same compose ergonomics, no throughput penalty.

### Linux/CUDA deployment (future migration, all-in-Docker)
When the gateway moves off the Mac mini to a dedicated GPU host, containers can access the GPU via NVIDIA Container Toolkit and vLLM runs inside Docker with full acceleration:

```
┌─── Linux host with RTX/A100/H100 ───┐
│  nvidia-container-toolkit           │
│  ┌──────────┐  ┌──────────┐         │
│  │ Caddy    │  │ gateway  │         │
│  └──────────┘  └──────────┘         │
│                │                    │
│  ┌─────────────▼────────────┐       │
│  │ vLLM container (GPU)     │       │
│  │ gemma4-31b-it            │       │
│  └──────────────────────────┘       │
└─────────────────────────────────────┘
```

For solo-Mac dev: drop Caddy, collapse to single gateway container + native MLX. Same code, different compose profile.

---

## Repo layout

```
gemma-gateway/
├── pyproject.toml           # uv / hatch, Python 3.12
├── Dockerfile               # multi-stage; runtime = python:3.12-slim
├── docker-compose.yml                      # Mac mini hybrid (default)
├── docker-compose.cpu-only.yml             # all-in-Docker fallback
├── docker-compose.cuda.yml                 # Linux GPU host
├── docker-compose.override.yml.example
├── Caddyfile
├── .env.example
├── .dockerignore
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py          # FastAPI app factory
│       ├── settings.py      # pydantic-settings
│       ├── auth.py          # API-key middleware
│       ├── rate_limit.py    # token bucket (Redis-backed)
│       ├── queue.py         # admission queue + priority
│       ├── circuit.py       # engine-side circuit breaker
│       ├── telemetry.py     # OTEL + Prometheus setup
│       ├── routers/
│       │   ├── generate.py
│       │   ├── embed.py
│       │   ├── rerank.py
│       │   ├── health.py
│       │   └── admin.py     # model list, reload, metrics helpers
│       ├── engines/
│       │   ├── base.py      # EngineProtocol
│       │   ├── vllm_engine.py
│       │   ├── mlx_engine.py
│       │   ├── ollama_engine.py
│       │   └── factory.py
│       ├── models/
│       │   ├── requests.py  # Pydantic
│       │   └── responses.py
│       └── errors.py
├── tests/
│   ├── unit/
│   ├── integration/         # real engine behind test fixture
│   ├── contract/            # OpenAPI drift tests
│   └── load/                # k6 / locust scripts
├── ops/
│   ├── grafana/             # dashboards JSON
│   ├── prometheus.yml
│   └── systemd/
│       └── gemma-gateway.service
└── PLAN.md
```

---

## API contract (v1)

All endpoints require `Authorization: Bearer <api-key>` except `/v1/health`.
Every request accepts/returns `X-Request-Id` for cross-service tracing.

### `POST /v1/generate`
```json
{
  "model": "gemma4-31b-it",
  "messages": [{"role":"user","content":"..."}],
  "temperature": 0.2,
  "max_tokens": 1024,
  "response_format": "text | json | tool_call",
  "json_schema": { "...optional JSON Schema (response_format=json)..." },
  "tools": [ { "name": "...", "description": "...", "parameters": {...} } ],
  "tool_choice": "auto | required | {\"name\":\"...\"}",
  "workflow": "trade_advisory",
  "priority": "normal | high",
  "stream": false
}
```
Native Gemma 4 function calling is surfaced via `tools` + `tool_choice`. The gateway normalizes engine-specific formats (vLLM, MLX, Ollama) into a consistent response shape.
Response:
```json
{
  "model": "gemma3-12b-it",
  "content": "...",
  "finish_reason": "stop | length | abstain | error",
  "tokens_in": 0,
  "tokens_out": 0,
  "latency_ms": 0,
  "request_id": "..."
}
```

### `POST /v1/embed`
Batch-first: accepts `inputs: string[]` up to configured max batch size. Returns vectors + model id + dims.

### `POST /v1/rerank`
`{ query, documents[], top_k }` → `[{ index, score }]`. Implemented only if `ENABLE_RERANK=true`.

### `GET /v1/health`
Liveness + readiness. Readiness checks: engine warm, queue depth under limit, Redis reachable (if configured).

### `GET /v1/models`
Lists loaded model ids, context window, embedding dims.

### `GET /metrics`
Prometheus scrape endpoint (scoped to admin network).

---

## High-traffic strategy

### 1. Continuous batching (the single biggest lever)
- **vLLM** natively continuously batches; target 16–64 concurrent sequences per GPU.
- **MLX-LM** on Apple Silicon supports batching via `mlx-lm.server`.
- Ollama is single-request by default — acceptable for dev, not for load.

### 2. Admission queue with priority
- Per-workflow priority (`high` for pre-trade review, `normal` for background summarization).
- Reject with **429** + `Retry-After` when queue depth exceeds threshold — do not silently buffer.

### 3. Rate limiting
- Token bucket per API key: requests-per-second and tokens-per-minute.
- Redis-backed when scaling horizontally; in-memory when single-process.

### 4. Horizontal scaling
- Multiple gateway replicas behind Caddy/Traefik.
- Multiple inference workers; gateway picks least-loaded via Redis-tracked queue depth.
- Model weights shared via a read-only mount so replicas start fast.

### 5. Caching
- **Response cache** keyed by `sha256(model + messages + temperature + json_schema)` with short TTL (5–60 min). Skip when `temperature > 0`.
- **Embedding cache** keyed by `sha256(model + text)` with long TTL (30 days). Huge win for re-embedding the same corpus chunks.
- Backing store: Redis (prod), sqlite (dev).

### 6. Streaming
- SSE for `/v1/generate` when `stream=true`.
- Lets client show partial output; frees server resources faster on client cancel.

### 7. Prompt caching (engine-side)
- vLLM supports prefix caching; enable for shared system prompts.
- MLX-LM has KV reuse helpers.

### 8. Backpressure and shedding
- Circuit breaker on engine: if engine error rate > 20% over 30s, shed non-`high` priority with 503.
- Queue TTL: requests older than `max_wait_ms` are dropped with 504 rather than processed late.

### 9. Timeouts everywhere
- Client-imposed max: reject if `timeout_ms > 60000`.
- Engine-side hard limit on `max_tokens`.
- Watchdog thread kills stuck generations.

---

## Capacity targets (v1, single host)

| Host class | Engine | Model | Realistic steady RPS | P95 latency (512 tok out) |
|---|---|---|---|---|
| **Mac mini M4 base 16–24GB** | MLX-LM | gemma4-e4b-it | 2–4 RPS | 1.5–3s |
| **Mac mini M4 base 32GB** | MLX-LM | gemma4-26b-q4 | 0.8–1.5 RPS | 7–14s |
| **Mac mini M4 Pro 48GB** | MLX-LM | gemma4-26b-q4 | 1.5–2.5 RPS | 5–10s |
| **Mac mini M4 Pro 64GB** | MLX-LM | gemma4-31b-q4 | 1.2–2 RPS | 6–12s |
| M2/M3 Max 64GB (MLX) | MLX-LM | gemma4-26b-q4 | 2–3 RPS | 5–10s |
| M3 Ultra 128GB+ (MLX) | MLX-LM | gemma4-31b-q4 | 3–5 RPS | 4–8s |
| RTX 4090 24GB (vLLM) | vLLM | gemma4-26b-awq | 12–20 RPS | 2.5–5s |
| A100 80GB (vLLM) | vLLM | gemma4-31b-bf16 | 35–55 RPS | 1.5–3s |
| H100 80GB (vLLM) | vLLM | gemma4-31b-bf16 | 70–110 RPS | 0.8–2s |

**Mac mini M4 reality:** the mini is a great dev and single-user host, but is not a high-traffic target. Plan to start there and migrate to a CUDA box once a second consumer app (ISA or AgentFlow) starts hitting the gateway concurrently.

"High traffic" is hardware-bound, not code-bound — the gateway will not turn a consumer Mac into a data center. For real production load, plan for at minimum a dedicated CUDA GPU host; Apple Silicon is great for dev and single-user workloads but not for multi-tenant bursts.

---

## Security

- API keys are random 32-byte URL-safe tokens, stored hashed (argon2) in a small sqlite/`api_keys.db`.
- Keys rotatable via `/v1/admin/keys` (admin API key only).
- TLS terminated at Caddy; gateway listens on localhost only.
- No prompt logging by default; `LOG_PROMPTS=false` is the default. When enabled for debugging, redaction list runs first.
- No egress in gateway process beyond model downloads and metrics push.
- CORS disabled by default; enable per-app origin explicitly.

---

## Observability

### Metrics (Prometheus)
- `gateway_requests_total{endpoint,status,workflow,api_key_id}`
- `gateway_latency_seconds{endpoint,workflow}` (histogram)
- `gateway_queue_depth{priority}`
- `gateway_tokens_total{direction,model}`
- `gateway_engine_errors_total{reason}`
- `gateway_circuit_state{state}`

### Traces (OpenTelemetry)
- Span per request with `workflow`, `model`, `api_key_id`, `queue_wait_ms`, `engine_ms`.
- Propagate `traceparent` from clients.

### Logs
- Structured JSON (loguru), `request_id` on every line. No prompt content by default.

### Dashboards
- Grafana JSON committed under `ops/grafana/` — traffic, latency, queue, errors, GPU/Neural Engine utilization.

---

## Testing

### Unit
- Request/response validation, schema-mode JSON enforcement, error mapping, auth, rate-limit math, circuit state machine.

### Integration
- Boot a real engine (Ollama in CI, smallest Gemma variant) and run a request matrix.
- Embedding determinism test (same input → same vector).
- JSON-schema adherence test (assert model output validates against provided schema).

### Contract
- Publish OpenAPI at `/openapi.json`; SMA and other consumers pin it.
- CI diff test fails on breaking schema changes without a version bump.

### Load
- `k6` scripts under `tests/load/`: steady-state, spike, soak.
- Assertions: P95 latency ceiling, error rate < 1%, no memory growth over 1h soak.

### Chaos
- Kill engine mid-request → gateway returns typed 503, never hangs.
- Redis outage → degrade to in-memory rate limiting, mark circuit half-open.

---

## Deployment

### Mac mini M4 hybrid (primary deployment)

**Step 1 — native engine on macOS (Metal-accelerated, not in Docker)**
```bash
# Option A: Ollama
brew install ollama
ollama pull gemma4:e4b            # verify exact tag on ollama.com/library/gemma4
brew services start ollama        # autostart on login
# exposes http://localhost:11434

# Option B: MLX-LM (preferred on M-series)
pip install mlx-lm
mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --port 11434
# wrap in LaunchAgent for autostart (see ops/launchd/)
```

**Step 2 — containerized services**
```bash
docker compose up -d
```
`docker-compose.yml` starts: gateway (FastAPI), Redis, Caddy, Prometheus, Grafana. Gateway reaches the native engine via `ENGINE_URL=http://host.docker.internal:11434`.

**Volumes**
- `gemma-gateway-redis-data` — request queue state, rate-limit counters.
- `gemma-gateway-prom-data` — 14-day metrics retention.
- `gemma-gateway-caddy-data` / `-config` — TLS certs, auto-renewed.
- `gemma-gateway-api-keys` — hashed API key store (sqlite).

**Networking**
- External: Caddy publishes `:443` (TLS). Gateway container is not exposed directly.
- Internal: `gemma-net` bridge connects gateway ↔ Redis ↔ Prometheus.
- Host access: gateway container uses `host.docker.internal` (Docker Desktop / OrbStack default) to reach the native MLX/Ollama process.

**Resource limits (docker-compose.yml)**
```yaml
services:
  gateway:
    deploy:
      resources:
        limits:   { cpus: "2.0", memory: "1g" }
        reservations: { memory: "256m" }
  redis:
    deploy:
      resources:
        limits:   { cpus: "0.5", memory: "256m" }
  prometheus:
    deploy:
      resources:
        limits:   { memory: "512m" }
```

**Docker Desktop VM memory**
Set Docker Desktop → Resources → Memory to **4 GB**. Gateway + Redis + Prom/Grafana + Caddy fit comfortably. The model stays in native macOS memory, outside the VM — giving Gemma full unified-memory access.

**Autostart on mac login**
LaunchAgent `com.gemma-gateway.compose.plist`:
```xml
<string>/bin/sh</string>
<string>-c</string>
<string>docker compose -f /Users/.../gemma-gateway/docker-compose.yml up -d</string>
```
Plus the native-engine LaunchAgent (MLX or Ollama) — both start at login, gateway health-probes engine before ready.

**Health and restart**
- Every container: `restart: unless-stopped`.
- Gateway healthcheck: `curl -f http://localhost:8080/v1/health`.
- If native engine is down, gateway stays up and returns typed `UNAVAILABLE` / `ABSTAIN` — never hangs.

### Dev-loop speed tip
Mount the source into the gateway container with `docker-compose.override.yml.example`:
```yaml
services:
  gateway:
    volumes: ["./src:/app/src"]
    command: uvicorn gateway.main:app --reload --host 0.0.0.0 --port 8080
```
Copy to `docker-compose.override.yml` for dev, leave out of version control.

### Fallback — all-in-Docker (CPU-only, not recommended)
```bash
docker compose -f docker-compose.cpu-only.yml up
```
Runs Ollama inside a container. Works, but inference drops to 8–15 tok/s on E4B, sub-2 tok/s on 26B. Use only for smoke tests on a machine without brew/pip access.

### Single-host prod (one workstation / one rented GPU box)
```
docker compose up -d
```
Starts: Caddy + 2 gateway replicas + vLLM worker + Redis + Prometheus + Grafana.

### Linux / CUDA host (future migration)
```bash
docker compose -f docker-compose.cuda.yml up -d
```
Adds nvidia-container-toolkit, runs vLLM inside a GPU-enabled container, mounts HF cache volume for weights. Same gateway image, different engine adapter.

### Bare-metal fallback (no Docker)
- `systemd` unit under `ops/systemd/gemma-gateway.service` runs uvicorn behind Caddy.
- Model weights cached under `/var/lib/gemma-gateway/models`.
- Used only if Docker is unavailable on the host.

### Kubernetes (future)
- Helm chart optional; deferred until we outgrow single-host.

---

## Rollout

1. **Week 1** — Scaffolding, settings, auth, `/v1/health`, Ollama engine. Local-only.
2. **Week 2** — `/v1/generate` + `/v1/embed`, rate limiting, admission queue, metrics.
3. **Week 3** — vLLM engine integration (or MLX-LM on Apple Silicon), streaming, prompt cache.
4. **Week 4** — Caddy, Redis, horizontal scaling test, load suite, first Grafana dashboard.
5. **Week 5** — Onboard SMA as first consumer (shadow mode). Gather real traffic metrics.
6. **Week 6** — Harden based on real data; add `/v1/rerank` if retrieval eval demands it.

Each week ends with a demo request from SMA and a Grafana screenshot.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Mac single-host can't handle multi-app concurrent load | Document capacity targets honestly; move to dedicated GPU box when usage justifies |
| Model output drift between Gemma versions breaks consumers | Pin `model` in each consumer config; add golden-output regression tests |
| Queue starvation under bursty loads | Priority lanes + age-based TTL drop |
| Secret leakage via logs | `LOG_PROMPTS=false` default + redaction filter + log review in CI |
| vLLM/MLX install pain | Document exact versions in `pyproject.toml`; ship Docker image for reproducibility |
| Redis becomes SPOF | Optional; gateway degrades gracefully to in-memory mode |

---

## Open questions
- Do we add OpenAI-compatible endpoints (`/v1/chat/completions`) as a v1.1 adapter for third-party tools? *Default: no, keep surface minimal.*
- Multi-tenant billing / cost accounting — deferred until a second external consumer exists.
- Fine-tuning / LoRA hosting — separate future plan.

---

## Mac mini M4 deployment profile (LOCKED — 16 GB base, Docker Desktop)

### Locked stack
- **Host**: Mac mini M4 base, 16 GB unified memory.
- **Docker runtime**: Docker Desktop ≥ 4.30 (per user choice).
- **Docker Desktop VM**: pinned to **2 GB memory, 2 CPUs** (reduced from default 4 GB to leave headroom for the native model and SMA workers).
- **Native engine (outside Docker)**: **Ollama** via `brew install ollama` + `brew services start ollama`. Binds `:11434`. MLX-LM deferred to week-3 perf tuning if needed.
- **Containerized services (v1)**: gateway (FastAPI) + Caddy + Redis. **No Prometheus, no Grafana in v1** — metrics write to a SQLite file; observability upgrade deferred until the gateway moves off this host.
- **Model**: `gemma4-e4b-it` quantized to Q4. Exact Ollama tag verified at `ollama pull` time against `ollama.com/library/gemma4`.
- **Context window**: 8K.
- **Embedding**: `nomic-embed-text-v1.5` via Ollama's native embedding endpoint.
- **Python tooling**: uv + Python 3.12, `pyproject.toml`-first.

### 16 GB steady-state memory budget
| Component | Target RAM |
|---|---|
| macOS + background apps | 3–4 GB |
| Docker Desktop VM (gateway + Caddy + Redis) | 2 GB |
| Native Ollama + `gemma4-e4b-it` | 3–4 GB |
| Postgres + SMA workers + Next.js | 2–3 GB |
| Browser + editor | 2–3 GB |
| **Total** | 12–16 GB, fits with minor swap at peak |

### v1 service map (trimmed)
| Service | State | Notes |
|---|---|---|
| gateway (FastAPI) | **IN** | single container, uvicorn, ~200 MB |
| Caddy | **IN** | TLS + reverse proxy, ~50 MB |
| Redis | **IN** | admission queue + rate limit, `--maxmemory 128mb` |
| Prometheus | **DEFERRED** | replaced by SQLite metrics store in gateway |
| Grafana | **DEFERRED** | add when moving off the mini |

### Startup order on the mini
1. LaunchAgent starts native Ollama (first run: 30–60s while it loads `gemma4-e4b-it`).
2. LaunchAgent runs `docker compose up -d` → gateway + Caddy + Redis come up.
3. Postgres + Timescale (SMA's DB) start from SMA's own compose or brew service.
4. Gateway healthcheck probes `host.docker.internal:11434`, enters ready state.
5. Consumer apps connect at `http://localhost:8080` (Caddy proxies) or `https://gemma-gateway.local` with a trust-store cert.

### Operational tuning
- `MLX_MAX_BATCH_SIZE=4` — above this, a base M4 mini falls behind.
- `MLX_KV_CACHE_LIMIT_MB=4096` — cap KV cache to avoid eviction thrash.
- macOS: set Mac mini to never sleep (`pmset -a sleep 0 disksleep 0`) while acting as gateway host.
- Two LaunchAgents: one for the native engine (MLX/Ollama), one for `docker compose up -d`. Both set `RunAtLoad=true`.
- Docker Desktop / OrbStack: pin **4 GB VM memory, 2 CPUs**. Don't over-provision — the model needs the host RAM, not the VM.
- Metric to watch: `gateway_queue_depth` — if it hits the rejection threshold under normal SMA traffic, you've outgrown the mini.
- If OrbStack: enable `Rosetta for x86/amd64 emulation` only if an image is arm64-unfriendly; prefer arm64-native images throughout.

### When to migrate off the mini
Any of:
- Sustained queue depth > 3 during business hours.
- P95 latency exceeds SMA advisory deadlines (currently 10s).
- A second consumer app starts generating real traffic.
- A model > 26B becomes the SMA default.

### Tier-up path
1. Mac mini M4 base 16/24GB → `gemma4-e4b-it`
2. Mac mini M4 base 32GB → `gemma4-26b-it` (Q4) at lower batch size
3. Mac mini M4 Pro 48GB → `gemma4-26b-it` (Q4) comfortably
4. Mac mini M4 Pro 64GB → `gemma4-31b-it` (Q4)
5. Dedicated RTX 4090 / A100 box → `gemma4-31b-it` (AWQ / bf16) via vLLM

Each tier is an env-var + `ollama pull` / MLX download change. No code change in consumer apps.

---

## Model notes (Gemma 4 — April 2026 release)

### Family
- **E2B / E4B** — edge (phones, Pi, Jetson). Near-zero-latency offline. Not targeted by this gateway.
- **26B** — workstation / consumer GPU (RTX 4090 class or 64GB Mac with q4).
- **31B** — workstation flagship; default model for this gateway. Benchmarks Google published at release: AIME 2026 math 89.2%, LiveCodeBench v6 80%.

### New capabilities worth wiring in v1
- **Native function calling** — surfaced as `tools` + `tool_choice` on `/v1/generate`. Replaces the brittle "prompt-plus-JSON-schema" pattern for structured outputs.
- **Multimodal (audio + vision)** — reserved for v1.1; add `images[]` + `audio[]` to `/v1/generate` once MLX/vLLM support is stable and a consumer app needs it.
- **140+ languages with cultural context** — already available; no gateway changes needed.

### Not provided by Gemma 4
- Embeddings. The gateway serves embeddings via a separate small model (`nomic-embed-text-v1.5` default, `bge-large-en-v1.5` alternate). `/v1/embed` is independent of the generation engine.
- Reranking. Optional `bge-reranker-v2-m3` behind `ENABLE_RERANK=true`.

### FunctionGemma (Dec 2025)
Google shipped FunctionGemma as a specialized edge function-calling sibling. Not used server-side in this gateway, but documented so consumer apps running on-device (AgentFlow) can pick it up independently.

### Ollama vs Google tag naming
Google publishes canonical sizes as E2B / E4B / 26B / 31B. Ollama's library uses its own tag scheme (`gemma4:27b`, `gemma4:12b`, etc.) that may not map 1:1. At deploy time, verify against `ollama.com/library/gemma4` and pin the gateway's `MODEL_TAG_MAP` accordingly. Consumer apps always send Google-canonical ids (`gemma4-31b-it`); the gateway translates.

### Model versioning discipline
- Consumer apps must pin `model` explicitly — never rely on `DEFAULT_MODEL`.
- Golden-output regression tests per consumer catch silent behavior changes across minor Gemma updates.
- Deprecation window: 60 days of overlap when bumping the default.
