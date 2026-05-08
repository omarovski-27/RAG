"""
eval/compare.py — Management KPI dashboard from logs/sessions.jsonl + turns.jsonl.

Usage
-----
  python -m eval.compare                        # all-time
  python -m eval.compare --since 2026-01-01    # from date onwards

Output (terminal)
-----------------
  Session KPIs table
  Per-topic retrieval frequency (from turns.jsonl)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
SESSIONS_FILE = ROOT / "logs" / "sessions.jsonl"
TURNS_FILE    = ROOT / "logs" / "turns.jsonl"

# Haiku pricing (per token)
COST_INPUT_PER_TOKEN  = 0.00000025   # $0.25 / 1M
COST_OUTPUT_PER_TOKEN = 0.00000125   # $1.25 / 1M


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _parse_date(s: str) -> datetime:
    """Parse ISO datetime string — strips microseconds-aware tz suffix."""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fallback for older Python
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _divz(num: float | int, den: float | int) -> float:
    return num / den if den else 0.0


def _pct(num: float | int, den: float | int) -> str:
    return f"{_divz(num, den):.1%}"


def _fmt_row(label: str, value: str, width: int = 34) -> str:
    return f"  {label:<{width}} {value}"


def run_compare(since: str | None = None) -> None:
    sessions = _load_jsonl(SESSIONS_FILE)
    turns    = _load_jsonl(TURNS_FILE)

    # ── Date filter ───────────────────────────────────────────────────────────
    if since:
        cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        sessions = [s for s in sessions if _parse_date(s["started_at"]) >= cutoff]
        turns    = [t for t in turns if _parse_date(t["timestamp"]) >= cutoff]
        filter_note = f" (since {since})"
    else:
        filter_note = " (all time)"

    if not sessions:
        print(f"No session records found in {SESSIONS_FILE}{filter_note}.")
        return

    # ── Session-level aggregates ──────────────────────────────────────────────
    total_sessions    = len(sessions)
    engaged           = [s for s in sessions if s.get("engaged", False)]
    total_engaged     = len(engaged)
    resolved          = sum(1 for s in engaged if s.get("outcome") == "resolved")
    escalated         = sum(1 for s in engaged if s.get("outcome") == "escalated")
    avg_turns         = _divz(sum(s.get("total_turns", 0) for s in engaged), total_engaged)

    # Latency: prefer per-turn records when available
    turn_latencies = [t["latency_ms"] for t in turns if t.get("latency_ms", 0) > 0]
    if turn_latencies:
        avg_latency = sum(turn_latencies) / len(turn_latencies)
    else:
        latencies_all = [s.get("avg_latency_ms", 0) for s in engaged if s.get("avg_latency_ms", 0) > 0]
        avg_latency = _divz(sum(latencies_all), len(latencies_all))

    # Tokens & cost (from turns when available, else from sessions)
    if any(t.get("total_tokens", 0) > 0 for t in turns):
        total_input  = sum(t.get("input_tokens", 0)  for t in turns)
        total_output = sum(t.get("output_tokens", 0) for t in turns)
    else:
        total_input  = sum(s.get("total_input_tokens",  0) for s in sessions)
        total_output = sum(s.get("total_output_tokens", 0) for s in sessions)
    total_tokens = total_input + total_output
    est_cost     = total_input * COST_INPUT_PER_TOKEN + total_output * COST_OUTPUT_PER_TOKEN

    # CSAT
    csat_ratings = [s["csat_rating"] for s in sessions if s.get("csat_rating") is not None]
    csat_avg     = _divz(sum(csat_ratings), len(csat_ratings))
    csat_rated   = len(csat_ratings)

    # Sentiment distribution (sum across all sessions that have it)
    sentiment_totals: Counter = Counter()
    for s in sessions:
        sc = s.get("sentiment_counts") or {}
        for k, v in sc.items():
            sentiment_totals[k] += v
    # Also count from turns for sessions that predate sentiment_counts
    for t in turns:
        sent = t.get("sentiment")
        if sent:
            sentiment_totals[sent] += 0  # ensure key exists; turns already covered by sessions

    # ── Print session KPI table ───────────────────────────────────────────────
    W = 60
    print(f"\n{'='*W}")
    print(f"  APG Chat - Management KPI Dashboard{filter_note}")
    print(f"{'='*W}")
    print(_fmt_row("Total sessions",              str(total_sessions)))
    print(_fmt_row("Engaged sessions",            str(total_engaged)))
    print(_fmt_row("Resolution rate",             _pct(resolved, total_engaged)))
    print(_fmt_row("Escalation rate",             _pct(escalated, total_engaged)))
    print(_fmt_row("Avg turns per session",       f"{avg_turns:.2f}"))
    print(_fmt_row("Avg latency per turn (ms)",   f"{avg_latency:.0f}"))
    print(_fmt_row("Total tokens",                f"{total_tokens:,}"))
    print(_fmt_row("  - input tokens",            f"{total_input:,}"))
    print(_fmt_row("  - output tokens",           f"{total_output:,}"))
    print(_fmt_row("Estimated API cost",          f"${est_cost:.6f}"))
    print(_fmt_row("CSAT avg",
                   f"{csat_avg:.2f}/5  (rated: {csat_rated}/{total_sessions})"))
    print(f"{'-'*W}")
    print("  Sentiment distribution:")
    for label in ("positive", "neutral", "frustrated", "angry"):
        count = sentiment_totals.get(label, 0)
        print(_fmt_row(f"    {label}", str(count)))
    print(f"{'='*W}")

    # ── Per-topic retrieval frequency (from turns.jsonl) ──────────────────────
    if not turns:
        print("  (No turns data available for topic breakdown)\n")
        return

    topic_counter: Counter = Counter()
    for t in turns:
        for topic in t.get("topics_retrieved", []):
            topic_counter[topic] += 1

    total_topic_hits = sum(topic_counter.values())
    print(f"\n  Per-topic retrieval frequency (turns.jsonl):")
    print(f"  {'Topic':<35}  {'Count':>5}  {'Share':>6}")
    print(f"  {'-'*35}  {'-'*5}  {'-'*6}")
    for topic, count in topic_counter.most_common():
        share = _pct(count, total_topic_hits)
        print(f"  {topic:<35}  {count:>5}  {share:>6}")
    print(f"  {'-'*35}  {'-'*5}")
    print(f"  {'Total':<35}  {total_topic_hits:>5}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG Chat KPI dashboard")
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Filter sessions starting on or after this date (UTC)",
    )
    args = parser.parse_args()
    run_compare(since=args.since)
