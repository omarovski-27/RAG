"""
memory.py — Conversation history management for the APG RAG chatbot.

Problem it solves
-----------------
Without memory every call to the chain is stateless. If a customer says:
  Turn 1: "What is the duty threshold for Saudi Arabia?"
  Turn 2: "What about the UAE?"           ← "What about" has no referent
  Turn 3: "How do I pay it?"              ← "it" refers to UAE duties

Claude can't answer turns 2 or 3 correctly without seeing the prior context.

How it works
------------
- One `ChatMessageHistory` object per session (keyed by session_id).
- Stored in a plain dict — in-memory, no DB needed for the PoC.
- `RunnableWithMessageHistory` wraps the RAG chain and automatically:
    1. Loads prior messages for this session_id before every invoke
    2. Appends the new Human + AI message pair after every invoke
- A sliding window (MAX_HISTORY_TURNS) keeps only the last N pairs so the
  context window doesn't grow unbounded. Default: 10 pairs = 20 messages.

Key functions
-------------
  get_session_history(session_id)  → ChatMessageHistory (singleton per session)
  build_memory_chain()             → RunnableWithMessageHistory
  ask_with_memory(question, session_id, retriever) → (answer, docs, latency_ms)
  clear_session(session_id)        → removes that session's history from memory

Usage (smoke test):
    python -m src.memory
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import LLM_MAX_TOKENS
from src.llm import get_llm
from src.prompts import get_rag_prompt_v2
from src.retriever import RerankingRetriever, get_reranking_retriever

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

# Number of prior Human+AI turn *pairs* to keep in the sliding window.
# 10 pairs = 20 messages = roughly 3-4k tokens of history overhead.
# Increase for longer support conversations; decrease to reduce token cost.
MAX_HISTORY_TURNS = 10

# ── In-memory store ─────────────────────────────────────────────────────────

# Dict keyed by session_id → ChatMessageHistory
# Lives for the lifetime of the process. app.py creates one session per
# Streamlit user session; they are independent.
_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Return the ChatMessageHistory for *session_id*, creating it if needed.
    RunnableWithMessageHistory calls this automatically — you rarely call
    it directly unless you need to inspect or clear the history.
    """
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session(session_id: str) -> None:
    """
    Remove all message history for *session_id*.
    Call this when a session ends so memory doesn't accumulate indefinitely.
    """
    _store.pop(session_id, None)


def _trim_history(session_id: str, max_pairs: int = MAX_HISTORY_TURNS) -> None:
    """
    Keep only the most recent *max_pairs* Human+AI message pairs.
    Called automatically by ask_with_memory after each turn.
    """
    history = get_session_history(session_id)
    messages = history.messages
    # Each pair = 2 messages (Human + AI). Keep the last max_pairs * 2.
    max_messages = max_pairs * 2
    if len(messages) > max_messages:
        history.messages = messages[-max_messages:]


# ── Memory-aware chain ───────────────────────────────────────────────────────

# Module-level cache — built once per process
_memory_chain: RunnableWithMessageHistory | None = None

# We keep a reference to the last retrieved docs so ask_with_memory can
# return them alongside the answer (needed by the logger).
_last_docs: List[Document] = []


def build_memory_chain(
    retriever: RerankingRetriever | None = None,
) -> RunnableWithMessageHistory:
    """
    Build a RAG chain wrapped with conversation memory.

    The chain input is a dict with two keys:
        {"question": str, "context": str}   ← context is pre-filled by us
    The chain output is a plain string (the answer).

    RunnableWithMessageHistory intercepts every invoke call:
      - Before: loads history[session_id] into the {history} slot
      - After:  appends the new Human + AI messages to history[session_id]

    input_messages_key  = "question"  (what to store as the Human message)
    history_messages_key = "history"  (the MessagesPlaceholder name in the prompt)
    """
    if retriever is None:
        retriever = get_reranking_retriever()

    prompt = get_rag_prompt_v2()
    llm = get_llm(streaming=False)

    # Core chain — context is passed in from outside (we retrieve first,
    # then call the chain) so the retriever result is also available to return.
    core_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    memory_chain = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return memory_chain


def get_memory_chain(
    retriever: RerankingRetriever | None = None,
) -> RunnableWithMessageHistory:
    """Singleton wrapper — builds the chain once per process."""
    global _memory_chain
    if _memory_chain is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            _memory_chain = build_memory_chain(retriever)
    return _memory_chain


def ask_with_memory(
    question: str,
    session_id: str,
    retriever: RerankingRetriever | None = None,
) -> Tuple[str, List[Document], float]:
    """
    Full memory-aware RAG pipeline.

    Flow:
      1. RerankingRetriever fetches + reranks chunks for *question*
      2. Chunks are formatted into a context string
      3. RunnableWithMessageHistory prepends conversation history + calls Claude
      4. History is updated automatically
      5. Sliding window trim keeps history bounded

    Returns
    -------
    answer        : Claude's response string
    retrieved_docs: the reranked chunks passed as context (for logger)
    latency_ms    : wall-clock time for the full call
    """
    if retriever is None:
        retriever = get_reranking_retriever()

    chain = get_memory_chain(retriever)

    t0 = time.perf_counter()

    # Step 1 — retrieve + rerank (outside the chain so we can return docs)
    docs = retriever.get_relevant_documents(question)
    context = _format_docs(docs)

    # Step 2 — invoke chain with history injection
    answer = chain.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Step 3 — trim history window
    _trim_history(session_id)

    return answer, docs, latency_ms


def _format_docs(docs: List[Document]) -> str:
    sections = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta.get('item_id', '?')}] {meta.get('item_title', '')}"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from src.logger import ConversationLogger
    from src.tools import detect_sentiment

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("ANTHROPIC_API_KEY not set — add it to .env and re-run.")
        raise SystemExit(0)

    # Simulate a 3-turn conversation where turns 2 and 3 rely on memory
    CONVERSATION = [
        "What is the duty threshold for Saudi Arabia?",
        "What about the UAE?",                          # "What about" — needs memory
        "How do I actually pay it?",                    # "it" = UAE duties — needs memory
    ]

    log = ConversationLogger()
    session_id = log.session_id
    print(f"Session: {session_id}\n")

    for turn_num, question in enumerate(CONVERSATION, 1):
        sentiment = detect_sentiment(question)
        print(f"Turn {turn_num} [{sentiment}]: {question}")

        answer, docs, latency = ask_with_memory(question, session_id)
        log.log_turn(question, docs, answer, latency_ms=latency, sentiment=sentiment)

        print(f"Answer : {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print(f"         [{len(docs)} chunks | {latency:.0f}ms]\n")

    # Check history is stored
    history = get_session_history(session_id)
    print(f"History: {len(history.messages)} messages stored for this session")

    summary = log.close(outcome="resolved", csat_rating=5)
    print(f"\nSession closed — turns: {summary['total_turns']}, "
          f"sentiment: {summary['sentiment_counts']}, "
          f"CSAT: {summary['csat_rating']}/5")

    # Clean up
    clear_session(session_id)
    print("Session history cleared from memory.")
