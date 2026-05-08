"""
eval/cost_report.py - Cost comparison: Copilot Studio vs Local RAG (Claude Haiku).

Reads actual token consumption from logs/sessions.jsonl.
Outputs a comparison table to terminal and eval/cost_report.txt.

Usage:
    python -m eval.cost_report
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT          = Path(__file__).parent.parent
SESSIONS_FILE = ROOT / "logs" / "sessions.jsonl"
TURNS_FILE    = ROOT / "logs" / "turns.jsonl"
REPORT_FILE   = Path(__file__).parent / "cost_report.txt"

# ── Haiku pricing ─────────────────────────────────────────────────────────────
HAIKU_INPUT_COST  = 0.00000025   # $0.25 per 1M input tokens
HAIKU_OUTPUT_COST = 0.00000125   # $1.25 per 1M output tokens
# Prompt caching: task spec says multiply total input cost by 0.30
# (70% cache hit rate; cached reads are 90% cheaper => effective multiplier ~0.30)
CACHE_MULTIPLIER  = 0.30
INFRA_MONTHLY     = 0.0          # running locally for demo

# ── Copilot Studio pricing ────────────────────────────────────────────────────
COPILOT_PER_TIER         = 200.0   # $ per tier
COPILOT_MSGS_PER_TIER    = 25_000  # messages per tier

# ── Volume assumption ─────────────────────────────────────────────────────────
VOLUME_MESSAGES = 25_000   # baseline monthly message / turn volume


# ── Data loading ──────────────────────────────────────────────────────────────

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


def _pull_token_stats(sessions: list[dict], turns: list[dict]) -> dict:
    """
    Return a dict with avg_input, avg_output, avg_turns, data_quality note.

    Strategy:
      1. Use sessions where total_tokens > 0 for per-session token averages.
         These are sessions where the logger captured token counts.
      2. If no session has token data (all streaming, none logged), fall back
         to estimating from actual turn answer lengths (4 chars ~ 1 token).
      3. avg_turns always uses all engaged sessions regardless of token data.
    """
    engaged = [s for s in sessions if s.get("engaged", True) and s.get("total_turns", 0) > 0]

    # avg turns per session (all engaged sessions)
    if engaged:
        avg_turns = sum(s["total_turns"] for s in engaged) / len(engaged)
    else:
        avg_turns = 1.0

    # sessions with real token data
    with_tokens = [s for s in engaged if s.get("total_tokens", 0) > 0]

    if with_tokens:
        avg_input  = sum(s.get("total_input_tokens",  0) for s in with_tokens) / len(with_tokens)
        avg_output = sum(s.get("total_output_tokens", 0) for s in with_tokens) / len(with_tokens)
        coverage   = len(with_tokens)
        total_sess = len(engaged)
        note = (
            f"Token data from {coverage}/{total_sess} sessions "
            f"(streaming sessions do not log tokens yet)."
        )
    else:
        # Fallback: estimate from actual answer character lengths in turns.jsonl
        # Typical RAG turn: ~1,200 input tokens (prompt + context + history)
        #                   ~120 output tokens (answer)
        # These are conservative estimates based on observed answer lengths.
        turns_with_answers = [t for t in turns if t.get("answer", "")]
        if turns_with_answers:
            avg_output_chars = sum(len(t["answer"]) for t in turns_with_answers) / len(turns_with_answers)
            est_output = avg_output_chars / 4.0   # ~4 chars per token
            est_input  = 1_200.0                  # system + 5 chunks + history + question
        else:
            est_input  = 1_200.0
            est_output = 120.0
        avg_input  = est_input  * avg_turns
        avg_output = est_output * avg_turns
        note = (
            "No token counts in sessions.jsonl (streaming sessions). "
            "Figures estimated from actual answer lengths in turns.jsonl."
        )

    return {
        "avg_input_per_session":  avg_input,
        "avg_output_per_session": avg_output,
        "avg_turns_per_session":  avg_turns,
        "data_note":              note,
        "sessions_total":         len(sessions),
        "sessions_engaged":       len(engaged),
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_usd(v: float) -> str:
    """Format a dollar amount. Tiny values keep extra decimal places."""
    if v == 0:
        return "$0.00"
    if v < 0.001:
        return f"${v:.6f}"
    if v < 1.0:
        return f"${v:.4f}"
    return f"${v:,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _row(label: str, copilot: str, local: str, w0: int, w1: int, w2: int) -> str:
    return f"| {label:<{w0}} | {copilot:>{w1}} | {local:>{w2}} |"


def _divider(w0: int, w1: int, w2: int) -> str:
    return f"+-{'-'*w0}-+-{'-'*w1}-+-{'-'*w2}-+"


# ── Main report ───────────────────────────────────────────────────────────────

def build_report() -> str:
    sessions = _load_jsonl(SESSIONS_FILE)
    turns    = _load_jsonl(TURNS_FILE)

    stats = _pull_token_stats(sessions, turns)

    avg_input  = stats["avg_input_per_session"]
    avg_output = stats["avg_output_per_session"]
    avg_turns  = stats["avg_turns_per_session"]

    # ── Local RAG cost per session ────────────────────────────────────────────
    input_cost_raw  = avg_input  * HAIKU_INPUT_COST
    input_cost_eff  = input_cost_raw * CACHE_MULTIPLIER   # with caching
    output_cost     = avg_output * HAIKU_OUTPUT_COST
    local_per_sess  = input_cost_eff + output_cost + INFRA_MONTHLY

    # ── Volume-based calculations ─────────────────────────────────────────────
    # Monthly sessions at 25,000 messages baseline
    sessions_per_month = VOLUME_MESSAGES / avg_turns

    # Copilot Studio: flat tier pricing
    tiers_current  = math.ceil(VOLUME_MESSAGES / COPILOT_MSGS_PER_TIER)
    tiers_2x       = math.ceil(2 * VOLUME_MESSAGES / COPILOT_MSGS_PER_TIER)
    copilot_monthly_curr = tiers_current * COPILOT_PER_TIER
    copilot_monthly_2x   = tiers_2x     * COPILOT_PER_TIER
    copilot_annual        = copilot_monthly_curr * 12

    # Copilot cost per session
    copilot_per_sess = copilot_monthly_curr / sessions_per_month

    # Local RAG monthly / annual
    local_monthly_curr = sessions_per_month * local_per_sess + INFRA_MONTHLY
    local_monthly_2x   = 2 * sessions_per_month * local_per_sess + INFRA_MONTHLY
    local_annual        = local_monthly_curr * 12

    # Savings
    annual_saving  = copilot_annual - local_annual
    saving_pct     = (annual_saving / copilot_annual * 100) if copilot_annual else 0.0

    # Cost per 1,000 sessions
    copilot_per_1k = copilot_per_sess * 1_000
    local_per_1k   = local_per_sess   * 1_000

    # ── Copilot pricing model string ──────────────────────────────────────────
    copilot_model_str = f"${COPILOT_PER_TIER:.0f} / {COPILOT_MSGS_PER_TIER//1000}k msgs (flat)"

    # ── Local RAG pricing model string ────────────────────────────────────────
    local_model_str = (
        f"Token-based, infra ${INFRA_MONTHLY:.0f}/mo"
    )

    # ── Table dimensions ──────────────────────────────────────────────────────
    w0, w1, w2 = 38, 24, 26
    div = _divider(w0, w1, w2)

    lines: list[str] = []

    def section(label: str, cop: str, loc: str) -> None:
        lines.append(_row(label, cop, loc, w0, w1, w2))

    lines.append("")
    lines.append("=" * (w0 + w1 + w2 + 10))
    lines.append("  APG Chat - Cost Comparison: Copilot Studio vs Local RAG")
    lines.append("=" * (w0 + w1 + w2 + 10))
    lines.append("")
    lines.append(f"  Sessions analysed      : {stats['sessions_total']}")
    lines.append(f"  Engaged sessions       : {stats['sessions_engaged']}")
    lines.append(f"  Avg turns / session    : {avg_turns:.2f}")
    lines.append(f"  Avg input tokens / ses : {avg_input:,.0f}")
    lines.append(f"  Avg output tokens / ses: {avg_output:,.0f}")
    lines.append(f"  Cache hit rate assumed : {CACHE_MULTIPLIER*100:.0f}% cost reduction on input")
    lines.append(f"  Baseline volume        : {VOLUME_MESSAGES:,} messages/month")
    lines.append(f"  Sessions at baseline   : {sessions_per_month:,.0f}/month")
    lines.append(f"  Note: {stats['data_note']}")
    lines.append("")
    lines.append(div)
    lines.append(_row("Metric", "Copilot Studio", "Local RAG (Claude Haiku)", w0, w1, w2))
    lines.append(div)
    section("Pricing model",
            copilot_model_str,
            local_model_str)
    lines.append(div)
    section("Cost per session",
            _fmt_usd(copilot_per_sess),
            _fmt_usd(local_per_sess))
    section("Cost per 1,000 sessions",
            _fmt_usd(copilot_per_1k),
            _fmt_usd(local_per_1k))
    lines.append(div)
    section(f"Monthly - current ({VOLUME_MESSAGES//1000}k msgs)",
            _fmt_usd(copilot_monthly_curr),
            _fmt_usd(local_monthly_curr))
    section(f"Monthly - 2x volume ({2*VOLUME_MESSAGES//1000}k msgs)",
            _fmt_usd(copilot_monthly_2x),
            _fmt_usd(local_monthly_2x))
    lines.append(div)
    section("Annual - current volume",
            _fmt_usd(copilot_annual),
            _fmt_usd(local_annual))
    section("Annual saving vs Copilot Studio",
            "--",
            _fmt_usd(annual_saving))
    section("Saving percentage",
            "--",
            _fmt_pct(saving_pct))
    lines.append(div)

    lines.append("")
    lines.append("  Assumptions:")
    lines.append(f"    Haiku input  : ${HAIKU_INPUT_COST*1_000_000:.2f} per 1M tokens")
    lines.append(f"    Haiku output : ${HAIKU_OUTPUT_COST*1_000_000:.2f} per 1M tokens")
    lines.append(f"    Cache saving : multiply input cost by {CACHE_MULTIPLIER} (70% hit rate, 90% cheaper)")
    lines.append(f"    Infrastructure: ${INFRA_MONTHLY:.0f}/month (local deployment)")
    lines.append(f"    Copilot Studio: ${COPILOT_PER_TIER:.0f} per {COPILOT_MSGS_PER_TIER:,} messages; flat tier per block")
    lines.append("")

    # One-line summary for slides
    summary = (
        f"At {VOLUME_MESSAGES:,} messages/month, Local RAG costs "
        f"{_fmt_usd(local_monthly_curr)}/month vs Copilot Studio's "
        f"{_fmt_usd(copilot_monthly_curr)}/month -- "
        f"saving {_fmt_usd(annual_saving)} per year ({saving_pct:.0f}% cheaper)."
    )
    lines.append("  SLIDE SUMMARY:")
    lines.append(f"  {summary}")
    lines.append("")
    lines.append("=" * (w0 + w1 + w2 + 10))
    lines.append("")

    return "\n".join(lines)


def run() -> None:
    report = build_report()
    print(report)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    run()
