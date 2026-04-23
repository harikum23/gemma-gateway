"""Documentation portal — public HTML page + localhost-only API key reveal."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(tags=["portal"])

# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>gemma-gateway docs</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.6;padding:24px 16px 64px}
.wrap{max-width:860px;margin:0 auto}
header{display:flex;align-items:center;gap:16px;padding:24px 0 32px;border-bottom:1px solid #2d3148}
header h1{font-size:1.6rem;font-weight:700;color:#f8fafc}
.badge-version{background:#1e2130;border:1px solid #3b4060;border-radius:6px;padding:2px 10px;font-size:0.8rem;color:#94a3b8}
.live-dot{display:inline-flex;align-items:center;gap:6px;font-size:0.85rem;color:#4ade80}
.live-dot::before{content:'';width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e}
section{margin-top:36px}
h2{font-size:1.1rem;font-weight:600;color:#cbd5e1;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #2d3148}
h3{font-size:0.95rem;font-weight:600;color:#94a3b8;margin:18px 0 6px}
.card{background:#1e2130;border:1px solid #2d3148;border-radius:10px;padding:20px 24px;margin-bottom:16px}
.base-url{font-family:monospace;font-size:1rem;color:#7dd3fc;background:#0f1117;padding:8px 14px;border-radius:6px;display:inline-block;margin-top:6px}
.method{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700;letter-spacing:.5px;margin-right:8px}
.GET{background:#1d4ed8;color:#bfdbfe}
.POST{background:#166534;color:#bbf7d0}
.DELETE{background:#7f1d1d;color:#fecaca}
.ADMIN{background:#4c1d95;color:#ddd6fe}
.ep-path{font-family:monospace;font-size:0.95rem;color:#f1f5f9;font-weight:600}
.ep-desc{color:#94a3b8;font-size:0.88rem;margin-top:4px}
details{margin-top:10px}
summary{cursor:pointer;color:#60a5fa;font-size:0.85rem;font-weight:500;user-select:none;outline:none;padding:4px 0}
summary:hover{color:#93c5fd}
pre{background:#0f1117;border:1px solid #2d3148;border-radius:6px;padding:14px 16px;font-size:0.78rem;color:#a5f3fc;overflow-x:auto;margin-top:8px;white-space:pre-wrap;word-break:break-all}
code{font-family:'Menlo','Consolas',monospace}
.ep-row{border-bottom:1px solid #2d3148;padding:14px 0}
.ep-row:last-child{border-bottom:none}
.auth-note{display:inline-block;font-size:0.78rem;padding:1px 8px;border-radius:4px;font-weight:600;margin-left:8px}
.auth-yes{background:#1e3a5f;color:#7dd3fc}
.auth-no{background:#1a2e1a;color:#86efac}
.auth-admin{background:#2e1a4a;color:#c4b5fd}
/* API key reveal */
.reveal-form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
.reveal-form input{background:#0f1117;border:1px solid #3b4060;border-radius:6px;color:#f1f5f9;padding:8px 12px;font-size:0.9rem;width:200px;outline:none}
.reveal-form input:focus{border-color:#3b82f6}
.reveal-form button{background:#3b82f6;border:none;border-radius:6px;color:#fff;padding:8px 18px;font-size:0.9rem;font-weight:600;cursor:pointer}
.reveal-form button:hover{background:#2563eb}
#key-result{margin-top:14px;display:none}
#key-box{font-family:monospace;font-size:0.88rem;background:#0f1117;border:1px solid #22c55e;border-radius:6px;padding:10px 14px;color:#4ade80;word-break:break-all}
#copy-btn{margin-top:8px;background:#166534;border:none;border-radius:6px;color:#bbf7d0;padding:6px 14px;font-size:0.82rem;cursor:pointer}
#copy-btn:hover{background:#15803d}
#key-error{color:#f87171;font-size:0.88rem;margin-top:8px;display:none}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>gemma-gateway</h1>
  <span class="badge-version">v0.1.0</span>
  <span class="live-dot">Live</span>
</header>

<section>
  <h2>Base URL</h2>
  <div class="card">
    <span class="base-url">http://localhost:8080</span>
  </div>
</section>

<section>
  <h2>Authentication</h2>
  <div class="card">
    <p>All endpoints except <code>/v1/health</code> require an API key passed as a Bearer token:</p>
    <pre><code>Authorization: Bearer &lt;api-key&gt;</code></pre>
    <p style="margin-top:10px;color:#94a3b8;font-size:0.88rem">Admin endpoints additionally require a key with <code>is_admin=true</code>. The <code>/v1/health</code> endpoint is always public.</p>
  </div>
</section>

<section>
  <h2>Endpoints</h2>

  <!-- /v1/health -->
  <div class="card">
    <div class="ep-row">
      <span class="method GET">GET</span>
      <span class="ep-path">/v1/health</span>
      <span class="auth-note auth-no">No Auth</span>
      <div class="ep-desc">Liveness and engine readiness check. Returns engine status, loaded model list, queue depth, and circuit breaker state.</div>
      <details><summary>Response schema</summary><pre><code>{
  "status":        "ok" | "degraded" | "down",
  "engine":        string,
  "engine_url":    string,
  "engine_ready":  boolean,
  "queue_depth":   integer,
  "queue_max":     integer,
  "circuit":       "closed" | "open" | "half-open",
  "models":        string[]
}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl http://localhost:8080/v1/health</code></pre></details>
    </div>
  </div>

  <!-- /v1/models -->
  <div class="card">
    <div class="ep-row">
      <span class="method GET">GET</span>
      <span class="ep-path">/v1/models</span>
      <span class="auth-note auth-yes">Bearer</span>
      <div class="ep-desc">List all model IDs currently loaded in the engine.</div>
      <details><summary>Response schema</summary><pre><code>{
  "models": [
    { "id": string, ...engine_metadata }
  ]
}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl http://localhost:8080/v1/models \\
  -H "Authorization: Bearer &lt;api-key&gt;"</code></pre></details>
    </div>
  </div>

  <!-- /v1/generate -->
  <div class="card">
    <div class="ep-row">
      <span class="method POST">POST</span>
      <span class="ep-path">/v1/generate</span>
      <span class="auth-note auth-yes">Bearer</span>
      <div class="ep-desc">Chat generation. Streams SSE by default (<code>stream=true</code>). Supports web search, function tools, structured JSON output, and workflow tagging.</div>
      <details><summary>Request schema</summary><pre><code>{
  "messages":          [{ "role": "user"|"assistant"|"system"|"tool", "content": string }],
  "model":             string | null,
  "temperature":       float (default 0.7),
  "max_tokens":        integer (default 512),
  "stop":              string[] | null,
  "stream":            boolean (default true),
  "enable_web_search": boolean (default false),
  "tools":             ToolDef[] | null,
  "tool_choice":       "auto"|"none"|"required" | null,
  "response_format":   "text" | "json_object" (default "text"),
  "json_schema":       object | null,
  "workflow":          string | null,
  "priority":          "normal" | "high" (default "normal"),
  "timeout_ms":        integer | null
}</code></pre></details>
      <details><summary>Response schema (non-streaming)</summary><pre><code>{
  "model":         string,
  "content":       string,
  "tool_calls":    [{ "name": string, "arguments": object }] | null,
  "finish_reason": "stop" | "length" | "tool_calls",
  "tokens_in":     integer,
  "tokens_out":    integer,
  "latency_ms":    float,
  "request_id":    string,
  "queue_wait_ms": float,
  "sources":       [{ "url": string, "title": string }] | null,
  "search_used":   boolean,
  "search_cache_hit": boolean
}</code></pre></details>
      <details><summary>SSE stream format</summary><pre><code>// delta events
data: {"delta": "&lt;token&gt;", "done": false, "request_id": "..."}

// final event
data: {"delta": "", "done": true, "content": "&lt;full&gt;", "model": "...", "request_id": "..."}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl -X POST http://localhost:8080/v1/generate \\
  -H "Authorization: Bearer &lt;api-key&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role":"user","content":"Hello!"}],
    "stream": false
  }'</code></pre></details>
    </div>
  </div>

  <!-- /v1/embed -->
  <div class="card">
    <div class="ep-row">
      <span class="method POST">POST</span>
      <span class="ep-path">/v1/embed</span>
      <span class="auth-note auth-yes">Bearer</span>
      <div class="ep-desc">Batch embeddings. Accepts up to 128 strings per request. Returns L2-normalized vectors by default.</div>
      <details><summary>Request schema</summary><pre><code>{
  "inputs":    string[],        // max 128 items, each max 32 000 chars
  "model":     string | null,
  "normalize": boolean (default true)
}</code></pre></details>
      <details><summary>Response schema</summary><pre><code>{
  "model":      string,
  "dims":       integer,
  "vectors":    float[][],
  "tokens_in":  integer,
  "latency_ms": float,
  "request_id": string
}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl -X POST http://localhost:8080/v1/embed \\
  -H "Authorization: Bearer &lt;api-key&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": ["hello world", "foo bar"]}'</code></pre></details>
    </div>
  </div>

  <!-- /v1/agent -->
  <div class="card">
    <div class="ep-row">
      <span class="method POST">POST</span>
      <span class="ep-path">/v1/agent</span>
      <span class="auth-note auth-yes">Bearer</span>
      <div class="ep-desc">Domain-agnostic ReAct agent loop. Built-in tools: <code>web_search</code>, <code>fetch_url</code>, <code>clean_text</code>, <code>dedupe</code>, <code>validate_schema</code>, <code>summarize</code>, <code>extract_entities</code>, <code>calculator</code>, <code>date_parse</code>, <code>translate</code>. Supports custom domain tools via a round-trip pause/resume flow.</div>
      <details><summary>Request schema</summary><pre><code>{
  "messages": [{ "role": "user"|"assistant"|"system"|"tool", "content": string }],
  "model":    string | null,
  "agent": {
    "max_steps":       integer (default 5, max 10),
    "total_budget_ms": integer | null,
    "tools":           string[],   // names of built-in tools to enable
    "domain_tools":    [{ "name": string, "description": string, "parameters": object }] | null
  }
}</code></pre></details>
      <details><summary>Response schema</summary><pre><code>{
  "content":    string,
  "steps":      integer,
  "tokens_in":  integer,
  "tokens_out": integer,
  "latency_ms": float,
  "request_id": string,
  // If a custom domain tool is requested, the agent pauses and returns:
  "status":     "paused",
  "tool_name":  string,
  "tool_args":  object
}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl -X POST http://localhost:8080/v1/agent \\
  -H "Authorization: Bearer &lt;api-key&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role":"user","content":"What is 42 * 99?"}],
    "agent": {"max_steps": 3, "tools": ["calculator"]}
  }'</code></pre></details>
    </div>
  </div>

  <!-- /v1/agent/tool_result -->
  <div class="card">
    <div class="ep-row">
      <span class="method POST">POST</span>
      <span class="ep-path">/v1/agent/tool_result</span>
      <span class="auth-note auth-yes">Bearer</span>
      <div class="ep-desc">Post back a custom domain tool result to resume a paused agent. Requires Redis. The <code>request_id</code> comes from the paused agent response.</div>
      <details><summary>Request schema</summary><pre><code>{
  "request_id": string,
  "tool_name":  string,
  "result":     string
}</code></pre></details>
      <details><summary>Response schema</summary><pre><code>// Same as /v1/agent response</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl -X POST http://localhost:8080/v1/agent/tool_result \\
  -H "Authorization: Bearer &lt;api-key&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{
    "request_id": "&lt;id-from-paused-response&gt;",
    "tool_name":  "my_lookup",
    "result":     "42"
  }'</code></pre></details>
    </div>
  </div>

  <!-- /v1/admin/keys -->
  <div class="card">
    <div class="ep-row">
      <span class="method POST">POST</span>
      <span class="ep-path">/v1/admin/keys</span>
      <span class="auth-note auth-admin">Admin</span>
      <div class="ep-desc">Create a new API key. The raw key is returned once — store it immediately.</div>
      <details><summary>Request schema</summary><pre><code>{
  "label":    string (default ""),
  "is_admin": boolean (default false)
}</code></pre></details>
      <details><summary>Response schema</summary><pre><code>{ "key": string }</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl -X POST http://localhost:8080/v1/admin/keys \\
  -H "Authorization: Bearer &lt;admin-key&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{"label": "my-service", "is_admin": false}'</code></pre></details>
    </div>
    <div class="ep-row">
      <span class="method GET">GET</span>
      <span class="ep-path">/v1/admin/keys</span>
      <span class="auth-note auth-admin">Admin</span>
      <div class="ep-desc">List all API keys (metadata only — hashes are never returned).</div>
      <details><summary>Response schema</summary><pre><code>{
  "keys": [
    {
      "key_id":      string,
      "label":       string,
      "is_admin":    boolean,
      "created_at":  string,
      "last_used_at": string | null,
      "revoked":     boolean
    }
  ]
}</code></pre></details>
      <details><summary>Example curl</summary><pre><code>curl http://localhost:8080/v1/admin/keys \\
  -H "Authorization: Bearer &lt;admin-key&gt;"</code></pre></details>
    </div>
  </div>

</section>

<section>
  <h2>API Key Reveal</h2>
  <div class="card">
    <p style="color:#94a3b8;font-size:0.88rem">Enter the operator passcode to reveal the active API key. Only works from localhost.</p>
    <div class="reveal-form">
      <input id="passcode" type="password" placeholder="Passcode" autocomplete="off"/>
      <button onclick="revealKey()">Reveal Key</button>
    </div>
    <div id="key-error">Invalid passcode</div>
    <div id="key-result">
      <div id="key-box"></div>
      <button id="copy-btn" onclick="copyKey()">Copy</button>
    </div>
  </div>
</section>

</div><!-- /wrap -->

<script>
async function revealKey() {
  var passcode = document.getElementById('passcode').value;
  var errEl = document.getElementById('key-error');
  var resultEl = document.getElementById('key-result');
  var boxEl = document.getElementById('key-box');
  errEl.style.display = 'none';
  resultEl.style.display = 'none';
  try {
    var resp = await fetch('/portal/apikey?passcode=' + encodeURIComponent(passcode));
    var data = await resp.json();
    if (!resp.ok || data.error) {
      errEl.style.display = 'block';
    } else {
      boxEl.textContent = data.api_key;
      resultEl.style.display = 'block';
    }
  } catch(e) {
    errEl.textContent = 'Request failed: ' + e.message;
    errEl.style.display = 'block';
  }
}
function copyKey() {
  var text = document.getElementById('key-box').textContent;
  navigator.clipboard.writeText(text).then(function() {
    document.getElementById('copy-btn').textContent = 'Copied!';
    setTimeout(function(){ document.getElementById('copy-btn').textContent = 'Copy'; }, 1500);
  });
}
document.getElementById('passcode').addEventListener('keydown', function(e){
  if (e.key === 'Enter') revealKey();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/portal",
    response_class=HTMLResponse,
    include_in_schema=False,
    summary="Documentation portal",
    description="Self-contained HTML documentation page for gemma-gateway. No auth required.",
)
async def portal() -> HTMLResponse:
    return HTMLResponse(content=_HTML)


@router.get(
    "/portal/apikey",
    summary="Reveal API key (localhost only)",
    description="Returns the first active API key when called from 127.0.0.1 with the correct passcode.",
)
async def portal_apikey(request: Request, passcode: str = "") -> JSONResponse:
    # Security note: the Docker port binding is 127.0.0.1:8080:8080, so only
    # host-local processes can reach this endpoint at all. Inside the container,
    # Docker NAT means request.client.host is the Docker bridge gateway IP
    # (172.x.x.x) rather than 127.0.0.1. We therefore accept loopback AND any
    # RFC-1918 address in the 172.16-31 and 10.x ranges (Docker bridge / overlay).
    # X-Forwarded-For from Caddy (port 8443) is also checked for the origin IP.
    import ipaddress

    def _is_local(ip_str: str) -> bool:
        if not ip_str:
            return False
        if ip_str in ("127.0.0.1", "::1", "localhost"):
            return True
        try:
            addr = ipaddress.ip_address(ip_str)
            return addr.is_loopback or addr.is_private
        except ValueError:
            return False

    direct_host = request.client.host if request.client else ""
    forwarded_for = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP", "").strip()

    if not any(_is_local(ip) for ip in (direct_host, forwarded_for, real_ip) if ip):
        return JSONResponse(status_code=403, content={"error": "forbidden"})

    if passcode != "0000":
        return JSONResponse(status_code=403, content={"error": "invalid passcode"})

    store = request.app.state.api_keys
    raw_keys: dict[str, str] = store.raw_keys

    # Prefer an active (non-revoked) key that was cached this session.
    if raw_keys:
        active_ids = {row["key_id"] for row in store.list_keys() if not row["revoked"]}
        for key_id, raw in raw_keys.items():
            if key_id in active_ids:
                return JSONResponse(content={"api_key": raw})
        # Fall through: all cached keys are revoked, generate a new one below.

    # raw_keys is empty (e.g. keys existed in DB from a prior run but their
    # raw values were never cached this session) — generate a fresh admin key
    # and cache it so subsequent calls also return it.
    new_raw = store.generate(label="portal-reveal", is_admin=True)
    return JSONResponse(content={"api_key": new_raw})
