"""
embedder.py — Embed KB Documents into FAISS indexes, one per topic.

Usage:
    python -m src.embedder          # build all indexes from kb/
    python -m src.embedder --rebuild # force-rebuild even if indexes exist
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, INDEXES_DIR, KB_DIR
from src.loader import load_all_kb_files

# Module-level singleton so the model is loaded once per process
_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.

    First call downloads the model (~130 MB) to ~/.cache/huggingface.
    Subsequent calls are instant.
    BGE models require a query instruction prefix for retrieval tasks;
    HuggingFaceEmbeddings handles this via encode_kwargs when we set
    the model_kwargs show_progress_bar flag.
    """
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def build_index_for_topic(
    documents: list,
    topic_name: str,
    output_dir: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Embed *documents* and persist a FAISS index to output_dir / topic_name /.

    Returns the path to the saved index directory.
    Overwrites an existing index when overwrite=True (default).
    """
    index_path = output_dir / topic_name

    if index_path.exists():
        if not overwrite:
            print(f"  [skip] {topic_name} — index already exists")
            return index_path
        shutil.rmtree(index_path)

    index_path.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()
    vectorstore = FAISS.from_documents(documents, embedder)
    vectorstore.save_local(str(index_path))

    print(f"  [ok]   {topic_name} — {len(documents)} vectors saved → {index_path}")
    return index_path


def build_all_indexes(
    kb_dir: Path = KB_DIR,
    indexes_dir: Path = INDEXES_DIR,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Load every .md file in kb_dir, build one FAISS index per topic, save to disk.

    Returns a dict mapping topic_name → index directory path.
    """
    indexes_dir.mkdir(parents=True, exist_ok=True)

    all_docs = load_all_kb_files(kb_dir)
    if not all_docs:
        raise RuntimeError(f"No KB files found in {kb_dir}")

    print(f"\nBuilding indexes for {len(all_docs)} topic(s) → {indexes_dir}\n")

    results: dict[str, Path] = {}
    for topic, docs in sorted(all_docs.items()):
        results[topic] = build_index_for_topic(
            docs, topic, indexes_dir, overwrite=overwrite
        )

    print(f"\nDone. {len(results)} index(es) written.\n")
    return results


def load_index(topic_name: str, indexes_dir: Path = INDEXES_DIR) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    index_path = indexes_dir / topic_name
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found for topic '{topic_name}' at {index_path}. "
            "Run `python -m src.embedder` first."
        )
    return FAISS.load_local(
        str(index_path),
        get_embedder(),
        allow_dangerous_deserialization=True,
    )


# ── CLI / smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS indexes from KB files.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if indexes already exist.",
    )
    args = parser.parse_args()

    build_all_indexes(overwrite=True)

    # Quick sanity-check: load the first index and run a similarity search
    from src.config import INDEXES_DIR as idx_dir

    indexes = list(idx_dir.iterdir()) if idx_dir.exists() else []
    if indexes:
        first_topic = indexes[0].name
        print(f"--- Sanity search on '{first_topic}' index ---")
        index = load_index(first_topic)
        hits = index.similarity_search("where is my order", k=3)
        for i, doc in enumerate(hits, 1):
            meta = doc.metadata
            print(
                f"  {i}. [{meta.get('item_id','?')}] {meta.get('item_title','?')}"
            )
