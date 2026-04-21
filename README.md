# gemma-gateway

High-traffic local LLM service exposing Gemma as LLM primitives (generate, embed, rerank) over HTTP.

See [`PLAN.md`](./PLAN.md) for full design.

## Quick start (Mac mini M4 hybrid — LOCKED v1)

```bash
# Prereqs: Docker Desktop running, Ollama native on :11434, gemma4:e4b pulled.
ollama list                       # verify gemma4:e4b is present
cp .env.example .env              # edit to set a bootstrap API key (optional)

docker compose up -d --build
docker compose logs -f gateway    # capture the auto-generated API key from first boot

curl -fsS http://localhost:8080/v1/health
curl -X POST http://localhost:8080/v1/generate \
  -H "Authorization: Bearer <API-KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Endpoints

| Path | Method | Auth | Notes |
|---|---|---|---|
| `/v1/health` | GET | no | Liveness + engine readiness |
| `/v1/models` | GET | yes | Loaded model ids |
| `/v1/generate` | POST | yes | Chat generation, streaming via `stream=true` |
| `/v1/embed` | POST | yes | Batch embeddings |
| `/v1/admin/keys` | POST/GET | admin | Rotate / list API keys |

## Development

```bash
uv sync --extra dev
uv run pytest
uv run uvicorn gateway.main:app --reload --port 8080
```
