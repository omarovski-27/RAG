"""
chain.py — RAG chain for the APG pipeline.

Two chains are available:

  build_simple_chain(topic)  — Phase 1, single-topic FAISS only. Kept for
                               per-topic debugging.

  build_rag_chain()          — Phase 2, full pipeline: EnsembleRetriever
                               (BM25 + all FAISS indexes) → CrossEncoder
                               reranker → Claude. This is what app.py uses.

Usage (smoke test — requires ANTHROPIC_API_KEY in .env):
    python -m src.chain
"""
import os
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import INDEXES_DIR
from src.embedder import load_index
from src.llm import get_llm
from src.prompts import get_rag_prompt_v1
from src.retriever import RerankingRetriever, get_reranking_retriever

load_dotenv()


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


# ── Phase 2: full hybrid chain ───────────────────────────────────────────────

def ask_full(
    question: str,
    retriever: RerankingRetriever | None = None,
) -> Tuple[str, List[Document], float]:
    """
    Full RAG pipeline: EnsembleRetriever → CrossEncoder reranker → Claude.

    Returns
    -------
    answer        : Claude's response string
    retrieved_docs: the reranked chunks that were passed as context
    latency_ms    : wall-clock time for the full pipeline

    This is the function called by app.py and by ConversationLogger.
    Retriever is a singleton by default — pass a custom one for testing.
    """
    if retriever is None:
        retriever = get_reranking_retriever()

    prompt = get_rag_prompt_v1()
    llm = get_llm(streaming=False)

    t0 = time.perf_counter()

    # Step 1 — retrieve + rerank
    docs = retriever.get_relevant_documents(question)

    # Step 2 — format context
    context = _format_docs(docs)

    # Step 3 — prompt → Claude → string
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = StrOutputParser().invoke(response)

    latency_ms = (time.perf_counter() - t0) * 1000

    return answer, docs, latency_ms


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your-key-here":
        print("ANTHROPIC_API_KEY not set — add it to .env and re-run.")
        raise SystemExit(0)

    from src.logger import ConversationLogger

    TEST_QUESTIONS = [
        "My parcel is held and the payment link is not working",
        "What is the duty threshold for Saudi Arabia?",
    ]

    log = ConversationLogger()
    print(f"Session: {log.session_id}\n")

    for q in TEST_QUESTIONS:
        print(f"Q: {q}")
        answer, docs, latency = ask_full(q)
        log.log_turn(q, docs, answer, latency_ms=latency)
        print(f"A: {answer[:200]}...")
        print(f"   [{len(docs)} chunks | {latency:.0f}ms | topics: {[d.metadata.get('topic') for d in docs]}]")
        print()

    summary = log.close(outcome="resolved")
    print("-" * 60)
    print(f"Session closed. Total turns: {summary['total_turns']}, tokens: {summary['total_tokens']}")
    print(f"Logs written to: logs/turns.jsonl + logs/sessions.jsonl")

