"""
eval/judge_prompts.py — LLM-as-judge scoring for APG RAG bot answers.

Scores each answer on three dimensions (0-2 each, total 0-6):
  factual_correctness : answer matches ground truth, correct facts/values
  groundedness        : no hallucinated facts beyond what retrieval supports
  tone                : professional, concise, APG persona, no hedging

Pass threshold: total_score >= 5

Usage (standalone):
    python -m eval.judge_prompts

Usage (imported):
    from eval.judge_prompts import judge_answer
    result = judge_answer(question, ground_truth, must_mention, bot_answer)
    # result["total_score"], result["pass"], result["factual_correctness"], ...
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).parent.parent

# Scoring rubric embedded in the prompt
_JUDGE_PROMPT = """\
You are evaluating a customer support bot's answer for APG, an international parcel logistics company.

QUESTION asked by the customer:
{question}

GROUND TRUTH SUMMARY (correct answer):
{ground_truth}

FACTS THAT MUST BE MENTIONED (if applicable to the answer):
{must_mention}

BOT ANSWER TO EVALUATE:
{bot_answer}

Score the bot answer on exactly three dimensions. For each dimension, assign a score of 0, 1, or 2:

factual_correctness (0-2):
  2 = All facts match ground truth; key values/rules are correct and present.
  1 = Mostly correct but one minor omission or slight inaccuracy.
  0 = Significant factual error, wrong values, or missing the core answer.

groundedness (0-2):
  2 = No claims beyond what the ground truth supports; no hallucinated facts.
  1 = One minor unsupported claim that doesn't mislead.
  0 = Hallucinated policy, invented numbers, or claims that contradict known facts.

tone (0-2):
  2 = Professional, concise, direct. No excessive hedging. Matches APG support persona.
  1 = Acceptable but slightly too verbose, vague, or informal.
  0 = Unprofessional, confusing, or so hedged it fails to answer clearly.

Respond with ONLY valid JSON. No text before or after the JSON block. Example format:
{{
  "factual_correctness": 2,
  "factual_correctness_reason": "One sentence explaining the score.",
  "groundedness": 2,
  "groundedness_reason": "One sentence explaining the score.",
  "tone": 1,
  "tone_reason": "One sentence explaining the score."
}}"""


def judge_answer(
    question: str,
    ground_truth_summary: str,
    must_mention: list[str],
    bot_answer: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Score a bot answer using Claude Haiku as judge.

    Parameters
    ----------
    question            : the original customer question
    ground_truth_summary: 1-2 sentence correct answer from questions.json
    must_mention        : list of facts that must appear in a correct answer
    bot_answer          : the answer produced by the RAG bot

    Returns
    -------
    dict with keys:
        factual_correctness, factual_correctness_reason,
        groundedness, groundedness_reason,
        tone, tone_reason,
        total_score (int 0-6),
        pass (bool, total_score >= 5)
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    must_mention_str = (
        "\n".join(f"- {m}" for m in must_mention)
        if must_mention
        else "(no specific required facts for this question)"
    )

    prompt_text = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth_summary,
        must_mention=must_mention_str,
        bot_answer=bot_answer,
    )

    llm = ChatAnthropic(
        model=model,
        temperature=0.0,
        max_tokens=512,
    )

    response = llm.invoke([HumanMessage(content=prompt_text)])
    raw = response.content.strip()

    # Extract JSON even if the model wraps it in markdown fences
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise ValueError(f"Judge returned non-JSON response: {raw[:200]}")

    scores = json.loads(json_match.group())

    total = (
        int(scores.get("factual_correctness", 0))
        + int(scores.get("groundedness", 0))
        + int(scores.get("tone", 0))
    )
    scores["total_score"] = total
    scores["pass"] = total >= 5
    return scores


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("ANTHROPIC_API_KEY not set — add it to .env and re-run.")
        raise SystemExit(0)

    test_question = "Can I pay duties with PayPal?"
    test_ground_truth = (
        "Only credit or debit card payment is accepted. "
        "PayPal is not accepted under any circumstances."
    )
    test_must_mention = ["PayPal is not accepted", "credit or debit card"]

    good_answer = (
        "Unfortunately PayPal is not accepted for duties payments under any circumstances. "
        "Only credit or debit card is accepted."
    )
    bad_answer = (
        "You can pay with PayPal, credit card, or bank transfer. "
        "All major payment methods are supported."
    )

    print("Testing judge on a GOOD answer:")
    result = judge_answer(test_question, test_ground_truth, test_must_mention, good_answer)
    print(json.dumps(result, indent=2))

    print("\nTesting judge on a BAD answer:")
    result = judge_answer(test_question, test_ground_truth, test_must_mention, bad_answer)
    print(json.dumps(result, indent=2))
