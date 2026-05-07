"""
prompts.py — System prompts and prompt templates for the APG RAG pipeline.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── System prompt (v1 — Phase 1, no tools, no memory) ──────────────────────
RAG_SYSTEM_PROMPT_V1 = """\
You are APG Chat, the customer support assistant for APG eCommerce.

Answer the user's question using ONLY the context provided below. \
Do not use prior knowledge or make up information.
If the context does not contain enough information to answer, say so clearly \
and direct the user to APG support: generalsupport@apgecommerce.com.

Tone: professional, friendly, concise. No hedging phrases like \
"I think", "might", or "possibly". State facts directly.

Context:
{context}
"""

# ── Chat prompt template (Phase 1 — no memory) ─────────────────────────────
def get_rag_prompt_v1() -> ChatPromptTemplate:
    """
    Return the Phase 1 prompt template.
    Slots: {context} (retrieved chunks), {question} (user input).
    No message history yet — that's added in Session 9.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT_V1),
            ("human", "{question}"),
        ]
    )
