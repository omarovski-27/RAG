"""
tools.py — LangChain tools available to Claude during a conversation.

Two tools are defined:

  get_tracking_status(tracking_number)
      Mock carrier lookup. Returns current status, carrier, and estimated
      delivery for a parcel. In production this would call the carrier API
      (DHL, Aramex, etc.). For the PoC it returns realistic randomised data
      so the demo feels live.

  escalate_to_human(reason)
      Signals that the conversation should be handed off to a human agent.
      Returns a structured dict that app.py renders as a handoff card and
      that ConversationLogger records as tool_called="escalate_to_human".

Both are exposed as LangChain StructuredTool objects (APG_TOOLS list) so
they can be bound to the Claude LLM in app.py with llm.bind_tools(APG_TOOLS).

Adding a new tool:
  1. Write the plain Python function with type-annotated args.
  2. Wrap it with StructuredTool.from_function(...).
  3. Append to APG_TOOLS.
  Nothing else in the system needs to change.

Usage (smoke test):
    python -m src.tools
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ── Mock data ───────────────────────────────────────────────────────────────

_CARRIERS = ["DHL", "Aramex", "FedEx", "UPS", "SMSA"]

_STATUSES = [
    "In Transit",
    "Out for Delivery",
    "Delivered",
    "Held at Customs",
    "Pending Duty Payment",
    "Shipment Created",
    "Returned to Sender",
]

# Statuses that map naturally to a "held / duty payment" context
_DUTY_STATUSES = {"Held at Customs", "Pending Duty Payment"}


def _seed_from_tracking(tracking_number: str) -> int:
    """Derive a stable integer seed from the tracking number string.
    Same tracking number always returns the same mock result."""
    return int(hashlib.md5(tracking_number.encode()).hexdigest(), 16) % (10 ** 8)


# ── Tool 1: get_tracking_status ─────────────────────────────────────────────

class TrackingInput(BaseModel):
    tracking_number: str = Field(
        description="The parcel tracking number, e.g. 1234567890 or JD014600006281471990"
    )


def get_tracking_status(tracking_number: str) -> dict:
    """
    Look up the current status of a parcel by tracking number.

    Returns a dict with:
      tracking_number : echoed back
      carrier         : carrier name (DHL, Aramex, FedEx, UPS, SMSA)
      status          : current tracking status
      last_update     : ISO date of last status change
      estimated_delivery : ISO date of estimated delivery (None if delivered/returned)
      requires_duty_payment : True if parcel is held pending duty payment
      message         : human-readable summary for the customer

    This is a mock implementation. In production, replace the body of this
    function with a real carrier API call. The return schema must stay the same.
    """
    rng = random.Random(_seed_from_tracking(tracking_number))
    carrier = rng.choice(_CARRIERS)
    status = rng.choice(_STATUSES)
    last_update = date.today() - timedelta(days=rng.randint(0, 3))
    requires_duty = status in _DUTY_STATUSES

    if status == "Delivered":
        estimated_delivery = None
        message = (
            f"Your parcel was delivered on {last_update.isoformat()}. "
            "If you have not received it, please contact us."
        )
    elif status == "Returned to Sender":
        estimated_delivery = None
        message = (
            "Your parcel is being returned to the sender. "
            "Please contact us to arrange re-delivery."
        )
    elif requires_duty:
        estimated_delivery = (last_update + timedelta(days=rng.randint(3, 7))).isoformat()
        message = (
            f"Your parcel is held at customs pending duty payment. "
            f"Please complete payment to release your shipment. "
            f"Estimated delivery after payment: {estimated_delivery}."
        )
    else:
        estimated_delivery = (last_update + timedelta(days=rng.randint(1, 5))).isoformat()
        message = (
            f"Your parcel is currently '{status}' with {carrier}. "
            f"Estimated delivery: {estimated_delivery}."
        )

    return {
        "tracking_number": tracking_number,
        "carrier": carrier,
        "status": status,
        "last_update": last_update.isoformat(),
        "estimated_delivery": estimated_delivery,
        "requires_duty_payment": requires_duty,
        "message": message,
    }


# ── Tool 2: escalate_to_human ────────────────────────────────────────────────

class EscalationInput(BaseModel):
    reason: str = Field(
        description=(
            "Brief reason why this conversation needs a human agent. "
            "E.g. 'Customer is distressed and demanding a refund' or "
            "'Issue is outside the scope of the knowledge base'."
        )
    )
    customer_sentiment: Optional[str] = Field(
        default="neutral",
        description="Detected sentiment: positive | neutral | frustrated | angry",
    )


def escalate_to_human(reason: str, customer_sentiment: str = "neutral") -> dict:
    """
    Signal that this conversation must be handed off to a human support agent.

    Claude should call this tool when:
      - The customer explicitly asks for a human / supervisor
      - The issue is outside the scope of the KB (damaged goods claim, refund, etc.)
      - The customer is repeatedly frustrated and the bot cannot resolve the issue
      - Any safety or legal concern is raised

    Returns a structured handoff record. app.py renders this as a handoff card.
    ConversationLogger records tool_called="escalate_to_human" for KPI tracking.
    """
    return {
        "tool": "escalate_to_human",
        "reason": reason,
        "customer_sentiment": customer_sentiment,
        "handoff_message": (
            "I'm connecting you with one of our support agents right now. "
            "They will have full context of our conversation. "
            "Please hold on — typical wait time is under 2 minutes."
        ),
        "action_required": "ROUTE_TO_HUMAN_QUEUE",
    }


# ── Sentiment detection ──────────────────────────────────────────────────────

# Keyword sets ordered from strongest signal to weakest.
# Rule-based is sufficient for a PoC — no extra model or latency.
# In production, swap the body of detect_sentiment() for a proper model call;
# the return values and the rest of the system stay the same.

_ANGRY_KEYWORDS = {
    "furious", "outraged", "outrageous", "unacceptable", "disgusting",
    "disgusted", "terrible", "awful", "horrible", "worst", "ridiculous",
    "incompetent", "scam", "fraud", "useless", "pathetic",
}

_FRUSTRATED_KEYWORDS = {
    "frustrated", "frustrating", "disappointed", "disappointing",
    "not working", "still waiting", "weeks", "never arrived", "nobody",
    "no response", "ignored", "wrong", "broken", "failed", "problem",
    "issue", "can't", "cannot", "why", "how long", "again",
}

_POSITIVE_KEYWORDS = {
    "thank", "thanks", "thank you", "great", "excellent", "perfect",
    "happy", "good", "appreciate", "helpful", "resolved", "solved",
    "wonderful", "amazing", "love", "pleased",
}

SentimentLabel = Literal["positive", "neutral", "frustrated", "angry"]


def detect_sentiment(text: str) -> SentimentLabel:
    """
    Classify the sentiment of a customer message.

    Returns one of: positive | neutral | frustrated | angry

    Precedence: angry > frustrated > positive > neutral
    This is intentionally simple for the PoC — replace the body with a
    model call (e.g. a HuggingFace zero-shot classifier) for production.
    """
    lower = text.lower()

    if any(kw in lower for kw in _ANGRY_KEYWORDS):
        return "angry"
    if any(kw in lower for kw in _FRUSTRATED_KEYWORDS):
        return "frustrated"
    if any(kw in lower for kw in _POSITIVE_KEYWORDS):
        return "positive"
    return "neutral"


# ── LangChain tool wrappers ──────────────────────────────────────────────────

TRACKING_TOOL = StructuredTool.from_function(
    func=get_tracking_status,
    name="get_tracking_status",
    description=(
        "Look up the live tracking status of a customer's parcel. "
        "Use this whenever the customer asks where their parcel is, "
        "what the tracking status means, or when it will be delivered. "
        "Input: the tracking number as a string."
    ),
    args_schema=TrackingInput,
)

ESCALATION_TOOL = StructuredTool.from_function(
    func=escalate_to_human,
    name="escalate_to_human",
    description=(
        "Hand off the conversation to a human support agent. "
        "Use this when: the customer asks for a human or supervisor; "
        "the issue cannot be resolved with the knowledge base; "
        "the customer is repeatedly frustrated; "
        "there is a refund, legal, or safety concern."
    ),
    args_schema=EscalationInput,
)

# All tools in one list — bind to Claude with: llm.bind_tools(APG_TOOLS)
APG_TOOLS = [TRACKING_TOOL, ESCALATION_TOOL]


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Tool 1: get_tracking_status")
    print("=" * 60)

    test_numbers = ["1234567890", "JD014600006281471990", "APG-99887766"]
    for tn in test_numbers:
        result = get_tracking_status(tn)
        print(f"\nTracking: {tn}")
        print(f"  Carrier : {result['carrier']}")
        print(f"  Status  : {result['status']}")
        print(f"  Duty?   : {result['requires_duty_payment']}")
        print(f"  Message : {result['message']}")

    print("\n" + "=" * 60)
    print("Tool 2: escalate_to_human")
    print("=" * 60)

    result = escalate_to_human(
        reason="Customer is asking for a full refund on a delivered parcel",
        customer_sentiment="frustrated",
    )
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"APG_TOOLS registered: {[t.name for t in APG_TOOLS]}")

    print("\n" + "=" * 60)
    print("Sentiment detection")
    print("=" * 60)
    samples = [
        "Thank you, that was very helpful!",
        "Where is my parcel?",
        "This is absolutely unacceptable, I've been waiting 3 weeks",
        "I'm disappointed, nobody has responded to my issue",
    ]
    for s in samples:
        print(f"  {detect_sentiment(s):>12}  →  {s}")

    print("\nSmoke test passed.")
