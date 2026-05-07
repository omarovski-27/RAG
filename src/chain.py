"""
chain.py — RAG chain for the APG pipeline.

Phase 1 (this file): minimal single-topic chain — no reranker, no memory.
Later sessions will extend this with hybrid retrieval, tools, and memory.

Usage (smoke test — requires ANTHROPIC_API_KEY in .env):
    python -m src.chain
"""
import os
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import INDEXES_DIR
from src.embedder import load_index
from src.llm import get_llm
from src.prompts import get_rag_prompt_v1


def _format_docs(docs: list) -> str:
    """Concatenate retrieved Document page_content into a single context block."""
    sections = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta.get('item_id', '?')}] {meta.get('item_title', '')}"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_simple_chain(topic: str, k: int = 5):
    """
    Build a minimal RAG chain over a single topic's FAISS index.

    Pipeline:
        question → FAISS retriever (top-k) → format docs → prompt → LLM → string

    Returns a LangChain Runnable that accepts {"question": str}.
    Requires the index for *topic* to exist under INDEXES_DIR.
    """
    vectorstore = load_index(topic, INDEXES_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = get_rag_prompt_v1()
    llm = get_llm(streaming=False)

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_simple(question: str, topic: str, k: int = 5) -> str:
    """
    Convenience wrapper: build the chain for *topic* and invoke with *question*.
    Returns the answer as a plain string.
    """
    chain = build_simple_chain(topic, k=k)
    return chain.invoke(question)


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("ANTHROPIC_API_KEY not set — add it to .env and re-run.")
        raise SystemExit(0)

    question = "Where is my order and what does it mean when it says shipment created?"
    topic = "wismo"

    print(f"Topic   : {topic}")
    print(f"Question: {question}")
    print("-" * 60)

    answer = ask_simple(question, topic)
    print(answer)
    print("-" * 60)
    print("Smoke test passed.")
