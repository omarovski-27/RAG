"""
retriever.py — Hybrid retrieval + cross-encoder reranking for the APG pipeline.

Two-stage process:
  1. EnsembleRetriever: BM25 (keyword) + FAISS (semantic) across ALL topics in parallel
     → returns up to K_INITIAL_RERANK candidates via reciprocal rank fusion
  2. RerankingRetriever: cross-encoder scores every (query, chunk) pair
     → returns top K_FINAL chunks for the prompt

Usage (smoke test):
    python -m src.retriever
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import (
    BM25_WEIGHT,
    FAISS_WEIGHT,
    INDEXES_DIR,
    K_FINAL,
    K_INITIAL_RERANK,
    K_PER_RETRIEVER,
    KB_DIR,
    RERANKER_MODEL,
)
from src.embedder import get_embedder, load_index
from src.loader import load_all_kb_files

# Module-level singletons
_reranker: CrossEncoder | None = None
_ensemble: EnsembleRetriever | None = None
_reranking_retriever: "RerankingRetriever | None" = None


def get_reranker() -> CrossEncoder:
    """
    Return a singleton CrossEncoder instance.
    Downloads ms-marco-MiniLM-L-6-v2 (~85 MB) on first call, cached after.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def build_ensemble_retriever(
    kb_dir: Path = KB_DIR,
    indexes_dir: Path = INDEXES_DIR,
    bm25_weight: float = BM25_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT,
    k_per_retriever: int = K_PER_RETRIEVER,
) -> EnsembleRetriever:
    """
    Build an EnsembleRetriever that searches ALL topic indexes simultaneously.

    - BM25Retriever: built in-memory from all KB Documents. Catches exact token
      matches for proper nouns (ASOS, KSA, PayPal, Aramex, SAR, etc.) that the
      embedder may mis-rank.
    - FAISS retrievers: one per topic, loaded from disk. Catch paraphrases and
      semantically similar content even when the words differ.

    Results are merged via reciprocal rank fusion weighted 40% BM25 / 60% FAISS.
    Returns up to k_per_retriever * 2 candidates (deduplicated by content hash).
    """
    # ── Load all documents for BM25 (in-memory, keyword index) ────────────
    all_docs_by_topic = load_all_kb_files(kb_dir)
    all_docs: List[Document] = [
        doc for docs in all_docs_by_topic.values() for doc in docs
    ]

    bm25 = BM25Retriever.from_documents(all_docs, k=k_per_retriever)

    # ── Load all FAISS indexes and merge into one retriever ────────────────
    embedder = get_embedder()
    topic_dirs = [p for p in indexes_dir.iterdir() if p.is_dir()]

    if not topic_dirs:
        raise RuntimeError(
            f"No indexes found in {indexes_dir}. "
            "Run `python -m src.embedder` first."
        )

    # Merge all topic FAISS indexes into a single vectorstore
    merged_vs: FAISS | None = None
    for topic_dir in sorted(topic_dirs):
        vs = FAISS.load_local(
            str(topic_dir),
            embedder,
            allow_dangerous_deserialization=True,
        )
        if merged_vs is None:
            merged_vs = vs
        else:
            merged_vs.merge_from(vs)

    faiss_retriever = merged_vs.as_retriever(
        search_kwargs={"k": k_per_retriever}
    )

    # ── Combine ────────────────────────────────────────────────────────────
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[bm25_weight, faiss_weight],
    )
    return ensemble


def get_ensemble_retriever(
    kb_dir: Path = KB_DIR,
    indexes_dir: Path = INDEXES_DIR,
) -> EnsembleRetriever:
    """Singleton wrapper — builds the ensemble once per process."""
    global _ensemble
    if _ensemble is None:
        _ensemble = build_ensemble_retriever(kb_dir, indexes_dir)
    return _ensemble


class RerankingRetriever:
    """
    Wraps an EnsembleRetriever with a cross-encoder reranking step.

    Flow:
      1. EnsembleRetriever fetches up to k_initial candidates
      2. CrossEncoder scores each (query, chunk) pair together
         (more accurate than comparing vectors independently)
      3. Returns top k_final chunks sorted by reranker score
    """

    def __init__(
        self,
        base_retriever: EnsembleRetriever | None = None,
        k_initial: int = K_INITIAL_RERANK,
        k_final: int = K_FINAL,
    ):
        self.base_retriever = base_retriever or get_ensemble_retriever()
        self.k_initial = k_initial
        self.k_final = k_final
        self._reranker = get_reranker()

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1 — broad retrieval
        candidates = self.base_retriever.invoke(query)

        # Deduplicate by item_id (BM25 and FAISS may return the same Item)
        seen: set[str] = set()
        unique: List[Document] = []
        for doc in candidates:
            key = doc.metadata.get("item_id", doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        if not unique:
            return []

        # Cap at k_initial before sending to cross-encoder
        pool = unique[: self.k_initial]

        # Step 2 — cross-encoder reranking
        pairs = [[query, doc.page_content] for doc in pool]
        scores = self._reranker.predict(pairs)

        # Step 3 — sort by score, return top k_final
        ranked = sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.k_final]]

    # LangChain compatibility alias
    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def get_reranking_retriever() -> RerankingRetriever:
    """Singleton — returns the shared RerankingRetriever, building it once per process."""
    global _reranking_retriever
    if _reranking_retriever is None:
        _reranking_retriever = RerankingRetriever()
    return _reranking_retriever


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_QUERIES = [
        ("proper noun / BM25 test",   "What is the threshold for KSA?"),
        ("semantic test",              "I can't pay my duties, what are my options?"),
        ("cross-topic test",           "My parcel is held and the payment link is not working"),
    ]

    print("Loading ensemble retriever (BM25 + FAISS)...")
    retriever = get_reranking_retriever()
    print(f"Reranker model: {RERANKER_MODEL}\n")

    for label, query in TEST_QUERIES:
        print(f"── {label} ──")
        print(f"   Query: {query}")
        results = retriever.get_relevant_documents(query)
        for i, doc in enumerate(results, 1):
            m = doc.metadata
            print(f"   {i}. [{m.get('topic','?')}] [{m.get('item_id','?')}] {m.get('item_title','?')}")
        print()
