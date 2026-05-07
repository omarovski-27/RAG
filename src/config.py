"""
config.py — Central constants for the APG RAG pipeline.
All tunable values live here. Nothing else imports from .env directly.
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
KB_DIR      = ROOT_DIR / "kb"
INDEXES_DIR = ROOT_DIR / "indexes"
LOGS_DIR    = ROOT_DIR / "logs"

# ── Embedding model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── Reranker model ─────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM ────────────────────────────────────────────────────────────────────
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 1024

# ── Retrieval ──────────────────────────────────────────────────────────────
BM25_WEIGHT       = 0.4
FAISS_WEIGHT      = 0.6
K_PER_RETRIEVER   = 10   # candidates from each retriever before fusion
K_INITIAL_RERANK  = 20   # candidates sent to cross-encoder
K_FINAL           = 5    # chunks that go into the prompt
