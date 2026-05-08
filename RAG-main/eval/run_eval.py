"""
eval/run_eval.py — Retrieval evaluation for the APG RAG pipeline.

Metrics
-------
  Hit@5  : fraction of questions where at least one expected item appears in
            the top-5 reranked results (checks both topic and item_id).
  MRR    : mean reciprocal rank — average of 1/rank for each first correct hit
            (0 contribution when there is no hit).

Usage
-----
  python -m eval.run_eval                     # label = timestamp
  python -m eval.run_eval --label sprint_1    # named run

Output
------
  eval/results/<label>.csv   — one row per question
  Terminal                   — summary table + per-topic breakdown
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def _check_hit(
    expected_item_ids: list[str],
    expected_topic: str,
    retrieved_docs: list,
) -> tuple[bool, int | None]:
    """
    Return (hit, rank) where rank is 1-indexed position of first matching doc.
    A match requires both topic == expected_topic AND item_id in expected_item_ids.
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata
        if (
            meta.get("item_id") in expected_item_ids
            and meta.get("topic") == expected_topic
        ):
            return True, rank
    return False, None


def run_eval(label: str | None = None) -> None:
    # Import here so the module can be imported cheaply for testing
    import sys, os
    sys.path.insert(0, str(ROOT))

    from dotenv import load_dotenv
    load_dotenv()

    from src.retriever import get_reranking_retriever

    if label is None:
        label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = _load_questions()
    print(f"\n{'='*60}")
    print(f"APG Retrieval Eval — run: {label}")
    print(f"Questions: {len(questions)} | Loading retriever…")

    retriever = get_reranking_retriever()
    print("Retriever ready.\n")

    rows: list[dict] = []
    topic_stats: dict[str, dict] = {}

    for i, q in enumerate(questions, start=1):
        qid            = q["id"]
        question       = q["question"]
        expected_topic = q["expected_topic"]
        expected_ids   = q["expected_item_ids"]

        docs = retriever.get_relevant_documents(question)

        hit, rank = _check_hit(expected_ids, expected_topic, docs)
        rr = (1.0 / rank) if rank is not None else 0.0

        retrieved_ids    = [d.metadata.get("item_id", "?") for d in docs]
        retrieved_topics = [d.metadata.get("topic", "?") for d in docs]

        rows.append({
            "id":                 qid,
            "question":           question,
            "expected_topic":     expected_topic,
            "expected_item_ids":  "|".join(expected_ids),
            "retrieved_item_ids": "|".join(retrieved_ids),
            "retrieved_topics":   "|".join(retrieved_topics),
            "hit":                hit,
            "rank":               rank if rank is not None else "",
            "reciprocal_rank":    round(rr, 4),
        })

        # Per-topic accumulation
        if expected_topic not in topic_stats:
            topic_stats[expected_topic] = {"hits": 0, "total": 0}
        topic_stats[expected_topic]["total"] += 1
        if hit:
            topic_stats[expected_topic]["hits"] += 1

        # Progress line
        hit_symbol = "HIT " if hit else "MISS"
        rank_str   = f"rank={rank}" if rank else "miss"
        print(f"  [{i:02d}/{len(questions)}] {hit_symbol} {qid:12s}  {rank_str}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"{label}.csv"
    fieldnames = [
        "id", "question", "expected_topic", "expected_item_ids",
        "retrieved_item_ids", "retrieved_topics", "hit", "rank", "reciprocal_rank",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Compute summary stats ─────────────────────────────────────────────────
    total      = len(rows)
    hits       = sum(1 for r in rows if r["hit"])
    hit_rate   = hits / total
    mrr        = sum(r["reciprocal_rank"] for r in rows) / total

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Hit@5 : {hits}/{total}  ({hit_rate:.1%})")
    print(f"  MRR   : {mrr:.4f}")
    print(f"  CSV   : {csv_path}")
    print(f"\n  Per-topic breakdown:")
    print(f"  {'Topic':<35}  {'Hits':>4}  {'Total':>5}  {'Rate':>6}")
    print(f"  {'-'*35}  {'-'*4}  {'-'*5}  {'-'*6}")
    for topic, stats in sorted(topic_stats.items()):
        rate = stats["hits"] / stats["total"]
        print(f"  {topic:<35}  {stats['hits']:>4}  {stats['total']:>5}  {rate:>6.1%}")
    print(f"{'='*60}\n")

    # ── Print failures ────────────────────────────────────────────────────────
    failures = [r for r in rows if not r["hit"]]
    if failures:
        print(f"  Failed questions ({len(failures)}):")
        for r in failures:
            print(f"    {r['id']:12s}  expected_topic={r['expected_topic']}")
            print(f"               expected={r['expected_item_ids']}")
            print(f"               got      ={r['retrieved_item_ids']}")
            print()
    else:
        print("  All questions passed Hit@5!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APG retrieval eval")
    parser.add_argument("--label", default=None, help="Run label (default: timestamp)")
    args = parser.parse_args()
    run_eval(label=args.label)
