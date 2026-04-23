from __future__ import annotations

import re

# ─── Query classification ─────────────────────────────────────────────────────
# Cheap regex-based intent classifier. Runs on the last user message, picks the
# most-specific match first. Used to tailor the coordinator system prompt so
# Gemma knows which facets to focus on when it synthesises the final answer.

_STOCK_PATTERNS = [
    r"\b(stock|share|ticker|nasdaq|nyse|nifty|sensex|dow|s&p|sp500|bse|nse)\b",
    r"\b(bull|bear|bullish|bearish|long|short|hold|option|call\s+option|put\s+option|rsi|macd)\b",
    r"\bmarket\s+(cap|price|value|performance|movers)\b",
    r"\b(earnings|dividend|ipo|split|buyback)\b",
    r"\b[A-Z]{2,5}\s+(stock|share|price|ticker|earnings|quote)\b",
]

_NEWS_PATTERNS = [
    r"\b(today|tonight|latest|breaking|just\s+in|right\s+now|currently)\b",
    r"\b(news|update|headline|announce[ds]?|reported|happening)\b",
    r"\bthis\s+(morning|afternoon|evening|week|month|hour)\b",
    r"\brecent(ly)?\b",
    r"\b(what\s+happened|what('?s|\s+is)\s+going\s+on)\b",
]

_FACTUAL_PATTERNS = [
    r"^\s*(what|who|when|where|how\s+many|how\s+much|which)\b",
    r"\b(define|definition|meaning\s+of|explain|describe)\b",
]


def classify_query(text: str) -> str:
    """Classify user intent → one of: stock, news, factual, general."""
    if not text:
        return "general"
    t = text.lower()
    for p in _STOCK_PATTERNS:
        if re.search(p, t):
            return "stock"
    for p in _NEWS_PATTERNS:
        if re.search(p, t):
            return "news"
    for p in _FACTUAL_PATTERNS:
        if re.search(p, t):
            return "factual"
    return "general"


# ─── Coordinator system prompts ───────────────────────────────────────────────

_BASE = (
    "You are a coordinator agent. Your job: interpret the user's question, "
    "use tools intelligently, and produce a precise, sourced answer.\n\n"
    "WORKFLOW:\n"
    "1. Before calling web_search, rewrite the user's question as 3-6 specific "
    "keywords. Include proper nouns, ticker symbols, dates, and entity names. "
    "Example: 'How is NVIDIA doing today?' → query='NVIDIA NVDA stock price today'\n"
    "2. After receiving search results, extract specific facts (numbers, dates, "
    "names). Do NOT paraphrase the search result verbatim — synthesize.\n"
    "3. When citing, use [1], [2] etc matching the order of the sources list.\n"
    "4. Never warn the user about real-time data limits — you have web_search.\n"
    "5. If the first search is insufficient, search again with refined terms "
    "(max 2 searches unless genuinely needed).\n\n"
)

_TEMPLATES: dict[str, str] = {
    "stock": _BASE + (
        "QUERY TYPE: FINANCIAL / MARKET\n"
        "Focus on: current price, daily change (% and absolute), volume, "
        "market cap, 52-week range, P/E if available, and the main catalyst "
        "driving movement. Lead with a compact numerical summary, then context. "
        "Always include the ticker symbol and exchange. "
        "If price data looks stale in the search result, say so explicitly."
    ),
    "news": _BASE + (
        "QUERY TYPE: CURRENT EVENTS / NEWS\n"
        "Focus on: what happened, when, who's involved, why it matters. "
        "Prefer sources from the last 24-72 hours. If sources disagree, "
        "flag the disagreement. Lead with the newest confirmed fact."
    ),
    "factual": _BASE + (
        "QUERY TYPE: FACTUAL LOOKUP\n"
        "Lead with a direct one-sentence answer, then brief context and "
        "any important caveats. Prefer authoritative sources (Wikipedia, "
        "official sites, government, academic)."
    ),
    "general": _BASE + (
        "QUERY TYPE: GENERAL\n"
        "Decide whether web_search is actually needed. If the question is "
        "not time-sensitive and you know the answer confidently, answer "
        "directly without a tool call. Be concise and precise."
    ),
}


def coordinator_prompt(query_type: str, tool_names: list[str]) -> str:
    """Return the tailored coordinator prompt for this query type."""
    base = _TEMPLATES.get(query_type, _TEMPLATES["general"])
    return base + f"\n\nAVAILABLE TOOLS: {', '.join(tool_names)}"


# ─── Gemini output post-processing ────────────────────────────────────────────
# Gemini 2.5 Flash routinely prefixes its synthesis with meta-commentary
# ("Here is a summary..."), and appends disclaimers about data freshness.
# These dilute the signal when Gemma synthesises the user-facing answer, so
# we strip them before the observation is added to the conversation.

_PREAMBLES = [
    # Only strip when there is a colon separator — "Here is a summary: <content>"
    # Avoids consuming sentences like "Based on results, NVDA rose 2%." as preambles.
    r"^\s*here\s+(is|are)\s+(a\s+|the\s+)?(summary|breakdown|overview|analysis|rundown|quick\s+look)[^\n:]*:\s*",
    r"^\s*based\s+on\s+(the\s+)?(search\s+results?|available\s+information|provided\s+sources?)[^\n:]*:\s*",
    r"^\s*here('?s|\s+is)\s+what\s+i\s+(found|know|can\s+tell\s+you)[^\n:]*:\s*",
    r"^\s*according\s+to\s+(the\s+)?(search\s+results?|sources?|latest)[^\n:]*:\s*",
    r"^\s*okay[,!]?\s+",
    r"^\s*sure[,!]?\s+here[^\n:]*:\s*",
]

_DISCLAIMERS = [
    r"(?im)^\s*\*{0,2}(note|disclaimer|important|warning|caveat)\*{0,2}[:\s]+[^\n]*(verify|confirm|outdated|real.?time|may\s+change|subject\s+to\s+change|knowledge\s+cutoff|last\s+update|not\s+financial\s+advice)[^\n]*\n?",
    r"(?im)^\s*(please|you\s+should|i\s+recommend)\s+(verify|confirm|double.?check|consult)[^\n]*\n?",
    r"(?im)as\s+of\s+(my\s+)?(last\s+update|knowledge\s+cutoff|training\s+data)[^,\.\n]*[,\.\n]",
    r"(?im)this\s+information\s+(may\s+be\s+outdated|is\s+current\s+as\s+of|reflects)[^\n]*\n?",
    r"(?im)^\s*\*{0,2}(i\s+am|i'm)\s+an\s+(ai|language\s+model)[^\n]*\n?",
]


def clean_gemini_output(text: str) -> str:
    """Strip preambles and disclaimers from Gemini's synthesised text."""
    if not text:
        return text
    cleaned = text.strip()

    # Strip one preamble at the start (they rarely stack)
    for pat in _PREAMBLES:
        m = re.match(pat, cleaned, re.IGNORECASE)
        if m:
            cleaned = cleaned[m.end():]
            break

    # Strip disclaimers wherever they appear
    for pat in _DISCLAIMERS:
        cleaned = re.sub(pat, "", cleaned)

    # Collapse triple+ blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
