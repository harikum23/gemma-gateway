from __future__ import annotations

import pytest

from gateway.agent.coordinator import (
    classify_query,
    clean_gemini_output,
    coordinator_prompt,
)


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:

    def test_stock_ticker_keyword(self):
        assert classify_query("How is NVDA stock performing today?") == "stock"

    def test_stock_explicit_word(self):
        assert classify_query("What is the current share price of Apple?") == "stock"

    def test_stock_market_cap(self):
        assert classify_query("What is NVIDIA's market cap?") == "stock"

    def test_stock_earnings(self):
        assert classify_query("TSLA earnings report this quarter") == "stock"

    def test_stock_sensex(self):
        assert classify_query("How is Sensex performing today?") == "stock"

    def test_stock_nifty(self):
        assert classify_query("Nifty 50 latest levels") == "stock"

    def test_news_today(self):
        assert classify_query("What happened today in the tech world?") == "news"

    def test_news_latest(self):
        assert classify_query("Latest news on the Fed rate decision") == "news"

    def test_news_breaking(self):
        assert classify_query("Breaking: OpenAI announces new model") == "news"

    def test_news_recently(self):
        assert classify_query("What has Elon Musk said recently?") == "news"

    def test_factual_what_is(self):
        assert classify_query("What is the capital of France?") == "factual"

    def test_factual_who_is(self):
        assert classify_query("Who is the CEO of NVIDIA?") == "factual"

    def test_factual_how_many(self):
        assert classify_query("How many employees does Google have?") == "factual"

    def test_general_fallback(self):
        # "explain" matches the factual pattern — use something with no pattern match
        assert classify_query("Write a poem about dogs") == "general"

    def test_general_empty_string(self):
        assert classify_query("") == "general"

    def test_stock_takes_priority_over_news(self):
        # "today" is news pattern but "stock" should win as more specific
        assert classify_query("NVDA stock price today") == "stock"


# ---------------------------------------------------------------------------
# coordinator_prompt
# ---------------------------------------------------------------------------

class TestCoordinatorPrompt:

    def test_stock_prompt_mentions_price_and_volume(self):
        prompt = coordinator_prompt("stock", ["web_search"])
        assert "price" in prompt.lower()
        assert "volume" in prompt.lower()
        assert "ticker" in prompt.lower()

    def test_news_prompt_mentions_recency(self):
        prompt = coordinator_prompt("news", ["web_search"])
        assert "24" in prompt or "recent" in prompt.lower()

    def test_factual_prompt_mentions_direct_answer(self):
        prompt = coordinator_prompt("factual", ["web_search"])
        assert "direct" in prompt.lower() or "one sentence" in prompt.lower()

    def test_general_prompt_mentions_decision(self):
        prompt = coordinator_prompt("general", ["web_search"])
        assert "web_search" in prompt.lower()

    def test_tool_names_listed_in_prompt(self):
        prompt = coordinator_prompt("stock", ["web_search", "fetch_url"])
        assert "web_search" in prompt
        assert "fetch_url" in prompt

    def test_unknown_type_falls_back_to_general(self):
        prompt = coordinator_prompt("unknown_type", ["web_search"])
        # Should not raise and should return something useful
        assert len(prompt) > 50

    def test_prompt_instructs_query_rewriting(self):
        prompt = coordinator_prompt("stock", ["web_search"])
        assert "keyword" in prompt.lower() or "rewrite" in prompt.lower()

    def test_prompt_bans_apology_response(self):
        prompt = coordinator_prompt("stock", ["web_search"])
        # All templates must prohibit the "I don't have real-time access" apology
        assert "never warn" in prompt.lower() or "do not" in prompt.lower() or "don't" in prompt.lower()


# ---------------------------------------------------------------------------
# clean_gemini_output
# ---------------------------------------------------------------------------

class TestCleanGeminiOutput:

    def test_strips_here_is_a_summary_prefix(self):
        raw = "Here is a summary of NVIDIA's performance:\n\nNVDA is up 3.2% today."
        result = clean_gemini_output(raw)
        assert not result.startswith("Here")
        assert "NVDA is up 3.2%" in result

    def test_strips_based_on_search_results_prefix(self):
        # Colon form is the preamble — content follows the colon
        raw = "Based on the search results:\n\nNVDA closed at $875."
        result = clean_gemini_output(raw)
        assert not result.lower().startswith("based on")
        assert "$875" in result

    def test_preserves_based_on_as_sentence(self):
        # No colon = it's a content sentence, not a preamble — must be preserved
        raw = "Based on recent earnings, NVDA closed at $875."
        result = clean_gemini_output(raw)
        assert "$875" in result

    def test_strips_heres_what_i_found_prefix(self):
        # Colon after "found" means it's a preamble
        raw = "Here's what I found: NVDA is trading at $910."
        result = clean_gemini_output(raw)
        assert "NVDA is trading" in result

    def test_strips_according_to_prefix(self):
        # Colon form is the preamble
        raw = "According to the search results: the stock rose 2%."
        result = clean_gemini_output(raw)
        assert "stock rose 2%" in result

    def test_strips_note_disclaimer(self):
        raw = "NVDA is up 3%.\n\nNote: Please verify this information as it may be outdated.\n"
        result = clean_gemini_output(raw)
        assert "NVDA is up 3%" in result
        assert "verify" not in result.lower()

    def test_strips_please_verify_disclaimer(self):
        raw = "The price is $875.\nPlease verify this with a financial advisor."
        result = clean_gemini_output(raw)
        assert "$875" in result
        assert "verify" not in result.lower()

    def test_strips_ai_disclaimer(self):
        raw = "NVDA rose 4%.\nI'm an AI language model and cannot guarantee accuracy."
        result = clean_gemini_output(raw)
        assert "NVDA rose 4%" in result
        assert "language model" not in result.lower()

    def test_preserves_clean_factual_text(self):
        raw = "NVDA is trading at $875.50, up 3.2% with volume of 42M shares."
        result = clean_gemini_output(raw)
        assert result == raw

    def test_handles_empty_string(self):
        assert clean_gemini_output("") == ""

    def test_handles_whitespace_only(self):
        result = clean_gemini_output("   \n\n   ")
        assert result == ""

    def test_collapses_excessive_blank_lines(self):
        raw = "NVDA up 3%.\n\n\n\n\nVolume was high."
        result = clean_gemini_output(raw)
        assert "\n\n\n" not in result

    def test_preserves_multiple_paragraphs(self):
        raw = "NVDA rose 3% today.\n\nThe rally was driven by AI chip demand."
        result = clean_gemini_output(raw)
        assert "NVDA rose 3%" in result
        assert "AI chip demand" in result
