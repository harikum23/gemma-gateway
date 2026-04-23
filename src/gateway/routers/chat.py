from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(tags=["chat"])

_TOOL_ICONS: dict[str, str] = {
    "web_search": "🔍",
    "fetch_url": "🌐",
    "summarize": "📝",
    "extract_entities": "🏷️",
    "calculator": "🧮",
    "date_parse": "📅",
    "translate": "🌍",
    "clean_text": "🧹",
    "dedupe": "✂️",
    "validate_schema": "✅",
}

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>gemma-gateway chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--card:#1e2130;--card2:#252840;--border:#2a2d45;
  --accent:#6c63ff;--accent2:#00d4aa;--text:#e2e4f0;--muted:#8b8fa8;
  --user-bubble:#2a2d55;--ai-bubble:#1e2130;
  --green:#22c55e;--amber:#f59e0b;--red:#ef4444;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px}
body{display:flex;flex-direction:column;height:100vh}

/* Stats bar */
#stats-bar{
  display:flex;align-items:center;gap:16px;padding:10px 20px;
  background:var(--card);border-bottom:1px solid var(--border);
  flex-shrink:0;flex-wrap:wrap;
}
#stats-bar .brand{font-weight:700;font-size:15px;color:var(--text);margin-right:8px}
.stat-item{display:flex;align-items:center;gap:5px;color:var(--muted);font-size:12px}
.stat-item span.val{color:var(--text);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;flex-shrink:0}
.dot.amber{background:var(--amber)}
.dot.red{background:var(--red)}
.dot.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
#stats-bar .spacer{flex:1}

/* Messages */
#messages{
  flex:1;overflow-y:auto;padding:20px;
  display:flex;flex-direction:column;gap:14px;
}
#messages::-webkit-scrollbar{width:5px}
#messages::-webkit-scrollbar-track{background:transparent}
#messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

.msg{display:flex;flex-direction:column;max-width:720px}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.ai{align-self:flex-start;align-items:flex-start}

.bubble{
  padding:10px 14px;border-radius:12px;line-height:1.6;
  white-space:pre-wrap;word-break:break-word;max-width:600px;
}
.msg.user .bubble{background:var(--user-bubble);border-bottom-right-radius:3px}
.msg.ai .bubble{background:var(--ai-bubble);border:1px solid var(--border);border-bottom-left-radius:3px}
.msg.ai .bubble.streaming::after{
  content:'▌';animation:blink .8s step-end infinite;
  color:var(--accent);margin-left:2px;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* Tool step cards */
.tool-cards{display:flex;flex-direction:column;gap:6px;margin-bottom:8px;max-width:600px}
.tool-card{
  background:var(--card2);border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:8px;overflow:hidden;font-size:12px;
}
.tool-card.web_search{border-left-color:#3b82f6}
.tool-card.fetch_url{border-left-color:#10b981}
.tool-card.calculator{border-left-color:#f59e0b}
.tool-card-header{
  display:flex;align-items:center;gap:8px;padding:7px 10px;
  cursor:pointer;user-select:none;
}
.tool-card-header:hover{background:rgba(255,255,255,.04)}
.tool-icon{font-size:14px}
.tool-name{font-weight:600;color:var(--text)}
.tool-ms{color:var(--muted);margin-left:auto}
.tool-chevron{color:var(--muted);transition:transform .2s;font-size:10px}
.tool-card.open .tool-chevron{transform:rotate(90deg)}
.tool-body{display:none;padding:0 10px 10px;border-top:1px solid var(--border)}
.tool-card.open .tool-body{display:block}
.tool-section{margin-top:8px}
.tool-section-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.tool-section pre{
  background:var(--bg);border-radius:4px;padding:6px 8px;
  font-size:11px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;
  color:#a5b4fc;
}
.tool-obs{color:var(--text);line-height:1.5;font-size:12px}

/* Sources */
.sources{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.source-pill{
  background:var(--card2);border:1px solid var(--border);border-radius:20px;
  padding:3px 10px;font-size:11px;color:var(--accent2);
  text-decoration:none;white-space:nowrap;
}
.source-pill:hover{border-color:var(--accent2);background:#1e3a38}

/* Metadata bar */
.meta-bar{
  display:flex;align-items:center;gap:12px;margin-top:5px;
  font-size:11px;color:var(--muted);flex-wrap:wrap;
}
.finish-badge{
  padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600;
  background:#1a3326;color:var(--green);border:1px solid #166534;
}
.finish-badge.amber{background:#2a1f08;color:var(--amber);border-color:#78350f}
.finish-badge.red{background:#2a0f0f;color:var(--red);border-color:#7f1d1d}

/* Thinking */
.thinking{
  display:flex;align-items:center;gap:8px;padding:10px 14px;
  background:var(--ai-bubble);border:1px solid var(--border);
  border-radius:12px;color:var(--muted);font-size:12px;
}
.thinking-dots span{
  display:inline-block;width:5px;height:5px;border-radius:50%;
  background:var(--muted);margin:0 2px;
  animation:thinking-bounce .8s ease-in-out infinite;
}
.thinking-dots span:nth-child(2){animation-delay:.15s}
.thinking-dots span:nth-child(3){animation-delay:.3s}
@keyframes thinking-bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}

/* Input bar */
#input-bar{
  padding:14px 20px;background:var(--card);border-top:1px solid var(--border);
  flex-shrink:0;
}
.input-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.mode-btn{
  padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;
  border:1px solid var(--border);background:transparent;color:var(--muted);
  cursor:pointer;transition:all .15s;
}
.mode-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.web-btn{
  padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;
  border:1px solid var(--border);background:transparent;color:var(--muted);
  cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:4px;
}
.web-btn.active{background:#1e3a5f;color:#60a5fa;border-color:#3b82f6}
.input-row{display:flex;gap:8px;margin-top:8px;align-items:flex-end}
#input{
  flex:1;padding:10px 14px;background:var(--card2);border:1px solid var(--border);
  border-radius:10px;color:var(--text);font-size:14px;resize:none;
  font-family:inherit;line-height:1.5;min-height:42px;max-height:160px;
  outline:none;transition:border-color .15s;
}
#input:focus{border-color:var(--accent)}
#send-btn{
  padding:10px 18px;background:var(--accent);color:#fff;border:none;
  border-radius:10px;font-weight:600;cursor:pointer;font-size:14px;
  transition:opacity .15s;flex-shrink:0;
}
#send-btn:disabled{opacity:.4;cursor:not-allowed}
#auth-status{font-size:11px;color:var(--muted);margin-top:6px}

/* Empty state */
.empty-state{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;color:var(--muted);text-align:center;padding:40px;
}
.empty-state .logo{font-size:48px;margin-bottom:8px}
.empty-state h2{font-size:18px;color:var(--text)}
.empty-state p{font-size:13px;max-width:360px;line-height:1.6}

@media(max-width:600px){
  #messages{padding:12px}
  #input-bar{padding:10px 12px}
  .bubble{max-width:100%}
  .tool-cards{max-width:100%}
}
</style>
</head>
<body>

<div id="stats-bar">
  <span class="brand">gemma-gateway</span>
  <span class="stat-item"><span class="dot pulse" id="health-dot"></span> <span id="health-text">connecting…</span></span>
  <span class="spacer"></span>
  <span class="stat-item">req <span class="val" id="stat-req">—</span></span>
  <span class="stat-item">avg <span class="val" id="stat-lat">—</span></span>
  <span class="stat-item">tok↑ <span class="val" id="stat-tin">—</span></span>
  <span class="stat-item">tok↓ <span class="val" id="stat-tout">—</span></span>
</div>

<div id="messages">
  <div class="empty-state" id="empty-state">
    <div class="logo">💬</div>
    <h2>gemma-gateway chat</h2>
    <p>Talk to Gemma 4 with live streaming, tool steps, web search grounding, and real-time gateway stats.</p>
  </div>
</div>

<div id="input-bar">
  <div class="input-controls">
    <button class="mode-btn active" id="btn-chat" onclick="setMode('chat')">Chat</button>
    <button class="mode-btn" id="btn-agent" onclick="setMode('agent')">Agent</button>
    <button class="web-btn" id="btn-web" onclick="toggleWeb()">🔍 Web</button>
  </div>
  <div class="input-row">
    <textarea id="input" placeholder="Type a message…" rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="send-btn" onclick="send()">Send</button>
  </div>
  <div id="auth-status"></div>
</div>

<script>
let apiKey = null;
let mode = 'chat';
let webSearch = false;
let busy = false;

// ─── Auth ───────────────────────────────────────────────────────────────────
async function initAuth() {
  try {
    const r = await fetch('/portal/apikey?passcode=0000');
    if (r.ok) {
      const d = await r.json();
      apiKey = d.api_key;
      document.getElementById('auth-status').textContent = '✓ authenticated';
      document.getElementById('auth-status').style.color = 'var(--green)';
    } else {
      document.getElementById('auth-status').textContent = '⚠ auth failed — set passcode';
      document.getElementById('auth-status').style.color = 'var(--amber)';
    }
  } catch {
    document.getElementById('auth-status').textContent = '⚠ gateway unreachable';
    document.getElementById('auth-status').style.color = 'var(--red)';
  }
}

// ─── Stats ───────────────────────────────────────────────────────────────────
async function fetchStats() {
  try {
    const r = await fetch('/chat/stats');
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('stat-req').textContent = d.requests ?? '—';
    document.getElementById('stat-lat').textContent = d.avg_latency_ms ? Math.round(d.avg_latency_ms) + 'ms' : '—';
    document.getElementById('stat-tin').textContent = fmtTok(d.tokens_in);
    document.getElementById('stat-tout').textContent = fmtTok(d.tokens_out);
    const dot = document.getElementById('health-dot');
    const txt = document.getElementById('health-text');
    if (d.circuit === 'open') {
      dot.className = 'dot red pulse'; txt.textContent = 'Circuit open';
    } else if (d.engine_ready === false) {
      dot.className = 'dot amber pulse'; txt.textContent = 'Engine warming';
    } else {
      dot.className = 'dot green pulse'; txt.textContent = 'Live';
    }
  } catch {}
}
function fmtTok(n) {
  if (!n) return '—';
  return n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n);
}
fetchStats();
setInterval(fetchStats, 15000);

// ─── Mode / toggle ───────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('btn-chat').classList.toggle('active', m === 'chat');
  document.getElementById('btn-agent').classList.toggle('active', m === 'agent');
}
function toggleWeb() {
  webSearch = !webSearch;
  document.getElementById('btn-web').classList.toggle('active', webSearch);
}

// ─── UI helpers ───────────────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}
function scrollBottom() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}
function removeEmpty() {
  const e = document.getElementById('empty-state');
  if (e) e.remove();
}
function addUserBubble(text) {
  removeEmpty();
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  document.getElementById('messages').appendChild(div);
  scrollBottom();
}
function createAiBubble() {
  removeEmpty();
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.innerHTML = `<div class="tool-cards" id="tool-cards-${_bubbleId}"></div>
    <div class="bubble streaming" id="bubble-${_bubbleId}"></div>`;
  document.getElementById('messages').appendChild(div);
  scrollBottom();
  return _bubbleId++;
}
let _bubbleId = 0;
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function setBusy(v) {
  busy = v;
  document.getElementById('send-btn').disabled = v;
  document.getElementById('input').disabled = v;
}

// ─── Send ─────────────────────────────────────────────────────────────────────
const history = [];

async function send() {
  if (busy) return;
  const inp = document.getElementById('input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  inp.style.height = 'auto';
  addUserBubble(text);
  history.push({role:'user', content:text});
  setBusy(true);
  try {
    if (mode === 'agent') await sendAgent(text);
    else await sendChat(text);
  } finally {
    setBusy(false);
  }
}

// ─── Chat mode (streaming SSE) ────────────────────────────────────────────────
async function sendChat(text) {
  const id = createAiBubble();
  const bubble = document.getElementById(`bubble-${id}`);
  const t0 = Date.now();
  let content = '';
  let meta = null;

  const body = {
    model: null,
    messages: history.slice(-20),
    stream: true,
    enable_web_search: webSearch,
  };

  try {
    const resp = await fetch('/v1/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.text();
      bubble.classList.remove('streaming');
      bubble.textContent = `Error ${resp.status}: ${err}`;
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        try {
          const d = JSON.parse(line.slice(5).trim());
          if (d.done) {
            meta = d;
          } else if (d.delta) {
            content += d.delta;
            bubble.textContent = content;
            scrollBottom();
          }
        } catch {}
      }
    }
  } catch (e) {
    bubble.classList.remove('streaming');
    bubble.textContent = `Network error: ${e.message}`;
    return;
  }

  bubble.classList.remove('streaming');

  if (meta) {
    history.push({ role: 'assistant', content: meta.content ?? content });
    if (meta.sources && meta.sources.length > 0) renderSources(bubble.parentElement, meta.sources);
    renderMeta(bubble.parentElement, {
      latency_ms: Date.now() - t0,
      tokens_in: meta.tokens_in,
      tokens_out: meta.tokens_out,
      finish_reason: meta.finish_reason,
    });
  } else {
    history.push({ role: 'assistant', content });
  }
  scrollBottom();
}

// ─── Agent mode ───────────────────────────────────────────────────────────────
async function sendAgent(text) {
  const id = createAiBubble();
  const bubble = document.getElementById(`bubble-${id}`);
  const toolCards = document.getElementById(`tool-cards-${id}`);
  const t0 = Date.now();

  // Show thinking indicator
  bubble.classList.remove('streaming');
  bubble.innerHTML = `<span class="thinking"><span style="margin-right:6px">Thinking</span><span class="thinking-dots"><span></span><span></span><span></span></span></span>`;

  const builtinTools = ['web_search'];
  const body = {
    model: null,
    messages: history.slice(-20),
    agent: {
      builtin_tools: webSearch ? builtinTools : [],
      max_steps: 6,
      total_budget_ms: 45000,
      return_trace: true,
    },
  };

  let data = null;
  try {
    const resp = await fetch('/v1/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.text();
      bubble.textContent = `Error ${resp.status}: ${err}`;
      return;
    }
    data = await resp.json();
  } catch (e) {
    bubble.textContent = `Network error: ${e.message}`;
    return;
  }

  // Render tool step cards
  if (data.steps && data.steps.length > 0) {
    toolCards.innerHTML = '';
    for (const step of data.steps) renderToolCard(toolCards, step);
  }

  // Render answer
  bubble.textContent = data.content ?? '';

  history.push({ role: 'assistant', content: data.content ?? '' });

  if (data.sources && data.sources.length > 0) renderSources(bubble.parentElement, data.sources);
  renderMeta(bubble.parentElement, {
    latency_ms: data.budget_used_ms ?? (Date.now() - t0),
    tokens_in: data.tokens_in,
    tokens_out: data.tokens_out,
    finish_reason: data.finish_reason,
  });
  scrollBottom();
}

// ─── Tool step cards ──────────────────────────────────────────────────────────
const TOOL_ICONS = {
  web_search:'🔍', fetch_url:'🌐', summarize:'📝', extract_entities:'🏷️',
  calculator:'🧮', date_parse:'📅', translate:'🌍', clean_text:'🧹',
  dedupe:'✂️', validate_schema:'✅',
};

function renderToolCard(container, step) {
  const icon = TOOL_ICONS[step.tool_name] || '🔧';
  const card = document.createElement('div');
  card.className = `tool-card ${step.tool_name}`;
  card.innerHTML = `
    <div class="tool-card-header" onclick="this.parentElement.classList.toggle('open')">
      <span class="tool-icon">${icon}</span>
      <span class="tool-name">${esc(step.tool_name)}</span>
      ${step.latency_ms ? `<span class="tool-ms">${Math.round(step.latency_ms)}ms</span>` : ''}
      <span class="tool-chevron">▶</span>
    </div>
    <div class="tool-body">
      ${step.tool_args ? `<div class="tool-section"><div class="tool-section-label">Input</div><pre>${esc(JSON.stringify(step.tool_args, null, 2))}</pre></div>` : ''}
      ${step.observation ? `<div class="tool-section"><div class="tool-section-label">Result</div><div class="tool-obs">${esc(String(step.observation).slice(0, 800))}</div></div>` : ''}
    </div>`;
  container.appendChild(card);
}

// ─── Sources ──────────────────────────────────────────────────────────────────
function renderSources(msgEl, sources) {
  const div = document.createElement('div');
  div.className = 'sources';
  for (const s of sources) {
    const a = document.createElement('a');
    a.className = 'source-pill';
    a.href = s.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    try { a.textContent = new URL(s.url).hostname.replace('www.',''); } catch { a.textContent = s.url.slice(0,40); }
    div.appendChild(a);
  }
  msgEl.appendChild(div);
}

// ─── Metadata bar ─────────────────────────────────────────────────────────────
function renderMeta(msgEl, { latency_ms, tokens_in, tokens_out, finish_reason }) {
  const fr = finish_reason || 'stop';
  const cls = fr === 'stop' ? '' : (fr === 'error' ? 'red' : 'amber');
  const div = document.createElement('div');
  div.className = 'meta-bar';
  div.innerHTML = `
    ${latency_ms ? `<span>⏱ ${Math.round(latency_ms)}ms</span>` : ''}
    ${tokens_in ? `<span>📥 ${tokens_in}</span>` : ''}
    ${tokens_out ? `<span>📤 ${tokens_out}</span>` : ''}
    <span class="finish-badge ${cls}">${esc(fr)}</span>`;
  msgEl.appendChild(div);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
initAuth();
document.getElementById('input').focus();
</script>
</body>
</html>"""


@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui() -> HTMLResponse:
    return HTMLResponse(content=_HTML)


@router.get("/chat/stats")
async def chat_stats(request: Request) -> JSONResponse:
    metrics: object = getattr(request.app.state, "metrics", None)
    circuit: object = getattr(request.app.state, "circuit", None)
    engine: object = getattr(request.app.state, "engine", None)

    summary: dict = {}
    if metrics is not None:
        try:
            summary = metrics.summary(window_s=300)
        except Exception:
            pass

    circuit_state = "closed"
    if circuit is not None:
        try:
            circuit_state = circuit.state
        except Exception:
            pass

    engine_ready = True
    if engine is not None:
        try:
            engine_ready = not getattr(engine, "_failed", False)
        except Exception:
            pass

    return JSONResponse({
        **summary,
        "circuit": circuit_state,
        "engine_ready": engine_ready,
    })
