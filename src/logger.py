"""
logger.py — Structured analytics logging for the APG RAG chatbot.

Two append-only JSON Lines files are written to logs/:

  turns.jsonl   — one record per message turn (question + answer + context)
  sessions.jsonl — one record per conversation session (outcome + aggregates)

Design principles:
  - Every field needed for management KPIs is captured at write time.
  - Adding a new metric = log one more field; nothing else changes.
  - Pure stdlib (json, datetime, pathlib, uuid, time) — no extra dependencies.

Management metrics this data supports:
  ┌─────────────────────────────┬────────────────────────────────────────┐
  │ Metric                      │ Source                                 │
  ├─────────────────────────────┼────────────────────────────────────────┤
  │ Total sessions              │ COUNT(sessions.jsonl)                  │
  │ Engaged sessions            │ WHERE engaged=true                     │
  │ Unengaged sessions          │ WHERE engaged=false                    │
  │ Resolved                    │ WHERE outcome=resolved                 │
  │ Escalated                   │ WHERE outcome=escalated                │
  │ Abandoned                   │ WHERE outcome=abandoned                │
  │ Sessions per day            │ GROUP BY date(started_at)              │
  │ Avg sessions per day        │ AVG of sessions-per-day                │
  │ Turns per day               │ GROUP BY date in turns.jsonl           │
  │ Avg turns per session       │ AVG(total_turns) in sessions.jsonl     │
  │ Avg latency per turn        │ AVG(latency_ms) in turns.jsonl         │
  │ API token consumption       │ SUM(input_tokens + output_tokens)      │
  │ Topics most retrieved       │ GROUP BY topics_retrieved in turns     │
  │ Escalation rate             │ escalated / engaged                    │
  │ Resolution rate             │ resolved / engaged                     │
  └─────────────────────────────┴────────────────────────────────────────┘

Usage:
    from src.logger import ConversationLogger
    logger = ConversationLogger()
    logger.log_turn(question, chunks, answer, tool_called=None, latency_ms=120)
    logger.close(outcome="resolved")  # or escalated / abandoned / unengaged
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from langchain_core.documents import Document

from src.config import LOGS_DIR

# Valid session outcome values
Outcome = Literal["resolved", "escalated", "abandoned", "unengaged"]


class ConversationLogger:
    """
    One instance per user conversation session.

    Instantiate when the session starts; call log_turn() after every
    bot response; call close(outcome) when the session ends.

    All writes are append-only JSON Lines — safe for concurrent processes.
    """

    TURNS_FILE = "turns.jsonl"
    SESSIONS_FILE = "sessions.jsonl"

    def __init__(self, logs_dir: Path = LOGS_DIR):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.session_id: str = str(uuid.uuid4())
        self.started_at: str = _utcnow()
        self.turns: list[dict] = []

    # ── Public API ───────────────────────────────────────────────────────

    def log_turn(
        self,
        question: str,
        retrieved_docs: List[Document],
        answer: str,
        *,
        tool_called: Optional[str] = None,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> dict:
        """
        Record one question→answer turn.

        Parameters
        ----------
        question      : raw user message
        retrieved_docs: chunks returned by RerankingRetriever (already reranked)
        answer        : final string returned by Claude
        tool_called   : name of any tool invoked this turn, or None
        latency_ms    : wall-clock time from question receipt to answer ready
        input_tokens  : tokens sent to Claude (for cost tracking)
        output_tokens : tokens returned by Claude (for cost tracking)
        """
        turn_number = len(self.turns) + 1
        topics = _extract_topics(retrieved_docs)
        item_ids = _extract_item_ids(retrieved_docs)

        record: dict = {
            # ── identifiers ─────────────────────────────────────────
            "session_id":    self.session_id,
            "turn_number":   turn_number,
            "timestamp":     _utcnow(),
            # ── content ─────────────────────────────────────────────
            "question":      question,
            "answer":        answer,
            "tool_called":   tool_called,
            # ── retrieval ────────────────────────────────────────────
            "chunks_retrieved": len(retrieved_docs),
            "topics_retrieved": topics,      # e.g. ["wismo","how_to_pay"]
            "item_ids_retrieved": item_ids,  # e.g. ["Item_003","Item_005"]
            # ── performance ──────────────────────────────────────────
            "latency_ms":    round(latency_ms, 1),
            # ── cost / consumption ───────────────────────────────────
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  input_tokens + output_tokens,
        }

        self._append_jsonl(self.TURNS_FILE, record)
        self.turns.append(record)
        return record

    def close(self, outcome: Outcome) -> dict:
        """
        Write the session summary record and return it.

        Call this exactly once — when the conversation ends for any reason.
        outcome must be one of: resolved | escalated | abandoned | unengaged
        """
        engaged = len(self.turns) > 0
        total_turns = len(self.turns)
        total_latency = sum(t["latency_ms"] for t in self.turns)
        total_input_tokens = sum(t["input_tokens"] for t in self.turns)
        total_output_tokens = sum(t["output_tokens"] for t in self.turns)
        escalated_turn = next(
            (t["turn_number"] for t in self.turns if t.get("tool_called") == "escalate_to_human"),
            None,
        )
        all_topics: list[str] = []
        for t in self.turns:
            all_topics.extend(t.get("topics_retrieved", []))
        unique_topics = sorted(set(all_topics))

        record: dict = {
            # ── identifiers ─────────────────────────────────────────
            "session_id":        self.session_id,
            "started_at":        self.started_at,
            "ended_at":          _utcnow(),
            # ── engagement & outcome ─────────────────────────────────
            "engaged":           engaged,
            "outcome":           outcome,           # resolved/escalated/abandoned/unengaged
            "escalated_at_turn": escalated_turn,    # None if not escalated
            # ── volume ───────────────────────────────────────────────
            "total_turns":       total_turns,
            # ── performance ──────────────────────────────────────────
            "total_latency_ms":  round(total_latency, 1),
            "avg_latency_ms":    round(total_latency / total_turns, 1) if total_turns else 0,
            # ── cost / consumption ───────────────────────────────────
            "total_input_tokens":  total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens":        total_input_tokens + total_output_tokens,
            # ── topic coverage ───────────────────────────────────────
            "topics_touched":    unique_topics,
            "topic_count":       len(unique_topics),
        }

        self._append_jsonl(self.SESSIONS_FILE, record)
        return record

    # ── Internal helpers ─────────────────────────────────────────────────

    def _append_jsonl(self, filename: str, record: dict) -> None:
        path = self.logs_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Module-level helpers ────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_topics(docs: List[Document]) -> List[str]:
    seen: list[str] = []
    for doc in docs:
        t = doc.metadata.get("topic", "unknown")
        if t not in seen:
            seen.append(t)
    return seen


def _extract_item_ids(docs: List[Document]) -> List[str]:
    return [doc.metadata.get("item_id", "?") for doc in docs]


# ── Convenience timer context manager ──────────────────────────────────────

class timer:
    """
    Usage:
        with timer() as t:
            answer = chain.invoke(question)
        latency = t.elapsed_ms
    """
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain_core.documents import Document

    print("Writing test session to logs/ ...")

    # Fake docs to simulate retrieved chunks
    fake_docs = [
        Document(page_content="Tracking status: In transit", metadata={"topic": "wismo", "item_id": "Item_001", "item_title": "Tracking Statuses"}),
        Document(page_content="Payment link steps...",        metadata={"topic": "how_to_pay", "item_id": "Item_003", "item_title": "Payment Link Not Working"}),
    ]

    log = ConversationLogger()
    print(f"Session ID: {log.session_id}")

    # Turn 1
    with timer() as t:
        pass  # simulate work
    log.log_turn("Where is my order?", fake_docs[:1], "Your order is in transit.", latency_ms=t.elapsed_ms, input_tokens=120, output_tokens=45)

    # Turn 2
    log.log_turn("What if the payment link is broken?", fake_docs, "Try requesting a new payment link.", latency_ms=310.5, input_tokens=180, output_tokens=60)

    # Close session
    summary = log.close(outcome="resolved")

    print("\nSession summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nCheck logs/turns.jsonl and logs/sessions.jsonl")
