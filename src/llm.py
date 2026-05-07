"""
llm.py — Configured ChatAnthropic instance for the APG RAG pipeline.

Usage (smoke test — requires ANTHROPIC_API_KEY in .env):
    python -m src.llm
"""
import os

from langchain_anthropic import ChatAnthropic

from src.config import CLAUDE_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

# Module-level singleton — one instance per process
_llm: ChatAnthropic | None = None
_llm_streaming: ChatAnthropic | None = None


def get_llm(streaming: bool = False) -> ChatAnthropic:
    """
    Return a configured ChatAnthropic instance.

    Reads ANTHROPIC_API_KEY from the environment (loaded via config.py → dotenv).
    streaming=False  — for eval runs and CLI tests (blocks until full response).
    streaming=True   — for the Streamlit UI (yields tokens as they arrive).

    Both share the same model, temperature, and max_tokens from config.py.
    """
    global _llm, _llm_streaming

    if streaming:
        if _llm_streaming is None:
            _llm_streaming = ChatAnthropic(
                model=CLAUDE_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                streaming=True,
            )
        return _llm_streaming
    else:
        if _llm is None:
            _llm = ChatAnthropic(
                model=CLAUDE_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                streaming=False,
            )
        return _llm


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("ANTHROPIC_API_KEY not set — add it to .env and re-run.")
        raise SystemExit(0)

    print(f"Model : {CLAUDE_MODEL}")
    print(f"Temp  : {LLM_TEMPERATURE}  |  Max tokens: {LLM_MAX_TOKENS}")
    print("Calling API...")

    llm = get_llm(streaming=False)
    response = llm.invoke("What is 2 + 2? Reply with one word.")
    print(f"Response: {response.content}")
    print("Smoke test passed.")
