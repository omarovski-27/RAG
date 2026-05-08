# APG Local RAG Chatbot — The Project Bible

**Owner:** Omar Alouran
**Target audience for demo:** Ryan Muldoon (then Paulina if approved)
**Build window:** 14 days at ~4 hrs/day = ~56 hours
**Doc version:** v2 (replaces the original 6-week plan)
**Last reviewed:** Phase 0, before any code is written

---

## How to read this document

This is your reference manual for the next two weeks. You do not read it once and discard it. You open it at the start of every session, find the section for that session, and follow it. When something breaks and you have to pick up the next day, this is what tells you where you were.

Three rules for using it:

1. If a section is unclear, that's a defect in the document, not a defect in you. Flag it and we fix the doc before you proceed.
2. Every session has a checkpoint. If the checkpoint doesn't pass, you do not move on. Loop back, ask for help, fix it.
3. The plan can change. Reality wins. If a decision turns out to be wrong (e.g. retrieval underperforms), we re-plan that section in writing before you keep building.

---

## Table of contents

**Part I — Foundation**
1. What changed from the original plan
2. The mental model (read this before anything else)
3. Tech stack — every choice justified
4. Architecture — three views
5. Knowledge base strategy

**Part II — The Build**
6. Project structure (annotated)
7. Phase 0 — Planning checkpoint
8. Phase 1 — Core RAG pipeline (Days 1–4)
9. Phase 2 — Hybrid retrieval + reranking (Days 5–7)
10. Phase 3 — Tools, memory, streaming (Days 8–10)
11. Phase 4 — Eval, UI polish, demo prep (Days 11–14)

**Part III — The Demo**
12. Evaluation framework
13. draw.io diagram specifications
14. Cost model
15. Demo script (minute by minute)
16. Q&A prep — what Ryan will ask

**Part IV — Operations**
17. Claude Code playbook
18. Risks, kill criteria, and rollback
19. What this is NOT
20. Appendix — command cheat sheet

---

# Part I — Foundation

## 1. What changed from the original plan

The original plan was solid in spine but thin in execution. The improvements in this document are not stylistic. Each one is a concrete capability or risk reduction. The headline changes:

**Timeline compressed from 6 weeks to 14 days.** At your stated 4 hrs/day, six weeks is 168 hours for a PoC. That's not a PoC, that's a side-job. Fourteen days is 56 hours, which is the right scale.

**Topic router removed.** The original used an LLM-based router to pick which FAISS index to search. This adds latency, cost, and a failure mode — and it's unnecessary once you have hybrid search and a reranker. We will retrieve from all indexes in parallel, then let the reranker decide what's relevant. Less code, more accurate, faster.

**Hybrid retrieval added (BM25 + dense).** Pure semantic search is bad at proper nouns and codes. Your KB is full of these: "ASOS", "Aramex", "Saudi Arabia", "SAR 1,000", "PayPal". Without keyword matching, "I'm in Saudi Arabia" might retrieve generic threshold content instead of the SAR-specific item. BM25 fixes this. This is the single highest-leverage accuracy improvement.

**Reranking added.** After initial retrieval, a cross-encoder model re-scores the top candidates against the query for true relevance. Runs locally, free, ~50ms. Lifts answer quality more than any prompt tweak.

**Tool use added (Anthropic native).** The current Copilot Studio bot has tracking API + ticket creation. The original plan ignored these. We use Anthropic's tool-calling API to register `get_tracking_status` and `escalate_to_human` as tools the model can invoke. For the demo the tracking tool returns mocked data; in a real deployment it would call APG's tracking endpoint.

**Prompt caching added.** Claude's prompt caching gives a ~90% discount on the cached portion of a prompt and ~85% latency reduction on cache hits. Your system prompt + retrieved chunks are the same shape across sessions; cache them. This is free money you cannot leave on the table when making a cost case.

**Streaming added.** A 4-second wait for the first token in a live demo will kill the impression. Streaming makes it feel responsive even when the underlying call takes the same total time.

**Eval framework upgraded.** Manual scoring of 33 questions is not reproducible. We add an LLM-as-judge harness so you can re-run after every change and compare versions objectively.

**Observability added.** Structured JSON logs for every query (input, retrieved chunks, scores, final answer, tokens, latency). Lets you debug bad answers and demonstrate to Ryan that you have visibility into what the bot is doing — something Copilot Studio actively obscures.

**Claude Code playbook added.** The original mentioned Claude Code in passing. We use it as a real co-developer: scaffolding, generating tests, running evals, debugging environment issues.

**draw.io diagram specs detailed.** Layout, colour codes, level of detail, what each box should and should not contain.

**LangChain retained throughout.** Per your call. We will use it for document loading, text splitting, FAISS wrapping, ensemble retrieval, prompt templates, output parsing, message history, and chains. Where LangChain is doing real work (FAISS wrapper, EnsembleRetriever) we lean on it. Where it's just ceremony (wrapping a string in PromptTemplate when an f-string would do), we use it anyway because that's the call you made and there's CV value in fluency with it.

---

## 2. The mental model — read this before anything else

The whole system has five jobs. Get these clear before any code, because every line you write will be doing one of them and you should always know which.

**Job 1 — Embedding.** Convert text into a vector. Done by a HuggingFace model running locally on your laptop. Free. Deterministic. Same input → same output, always.

**Job 2 — Retrieval.** Given a vector (and the original text), find the most relevant chunks of your KB. Done in two stages: a fast first pass (BM25 + FAISS in parallel), then a slow accurate second pass (cross-encoder reranker). Both run locally. Free.

**Job 3 — Generation.** Given retrieved chunks plus the user's question, produce a grounded answer. Done by Claude Haiku 4.5 over the Anthropic API. Pay per token.

**Job 4 — Tools.** Things the model can ask the system to do that aren't generation: look up a tracking number, escalate a ticket. Tools are functions Claude can decide to call mid-conversation. Native Anthropic API feature.

**Job 5 — State.** Conversation memory (so turn 3 remembers turn 1), session metadata (escalation counter, current topic), and structured logs.

LangChain is not a sixth job. It's the wiring that connects these. When you read code and see `RunnableLambda` or `EnsembleRetriever`, those are wires. The actual work is always one of the five jobs above.

If you can label every function in your codebase with its job number, you understand the system. That's the bar.

---

## 3. Tech stack — every choice justified

The stack below is what you will install. For every entry I give you the role, why it's the right call here, and what we considered and rejected. Read this once. It earns you credibility with Ryan when he asks why you didn't use X.

| Layer | Tool | Role | Why this choice |
|---|---|---|---|
| Runtime | Python 3.11+ | Everything runs here | Standard for ML; current LTS-equivalent |
| Orchestration | LangChain (latest stable) | Pipeline glue | Per your decision; familiar pattern, recruiter-recognisable |
| Embedding model | BGE-small-en-v1.5 (HuggingFace) | Text → 384-dim vectors | ~5 percentage points better retrieval than MiniLM-L6-v2 on standard benchmarks for almost identical size and speed. Local, free. |
| Lexical retrieval | rank-bm25 (via LangChain BM25Retriever) | Keyword matching | Required for proper nouns the embedder will mis-match (country names, ASOS, currency codes) |
| Vector store | FAISS (CPU build) | Dense search index | Lightweight, no server, fits in memory for our KB size, well-supported by LangChain |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-score retrieved candidates | Local, free, ~50ms per query, accuracy lift bigger than any prompt tweak |
| LLM | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Answer generation + tool calling | Cheap, fast, supports tool use, prompt caching, and streaming |
| Tool calling | Anthropic native tool use | `get_tracking_status`, `escalate_to_human` | Replaces the Copilot Studio Power Automate flows |
| Caching | Anthropic prompt caching | Cache system prompt + retrieved chunks | ~90% discount on cached portion, demo-critical for cost story |
| UI | Streamlit | Chat front-end | Fastest path to a credible UI. Built-in chat primitives. Streaming-friendly. |
| Eval | Anthropic API (LLM-as-judge) + a small Python harness | Score answers vs ground truth | Reproducible, fast, no external service |
| Logs | Python logging + JSON to file | Observability | No external service. Greppable. Demo-able. |
| Env | python-dotenv | API keys | Standard. `.env` never committed. |
| Version control | Git + private GitHub repo | Code only, no KB files | KB stays local — APG IP, never leaves your machine |
| Diagrams | draw.io desktop | Architecture diagrams | Per your existing toolchain |
| AI co-dev | Claude Code (CLI) | Scaffolding, tests, debugging | See Section 17 |

### What we considered and rejected

**OpenAI API.** You have an Anthropic key. Switching adds zero capability and a bill.

**ChromaDB / Qdrant / Weaviate.** Better databases than FAISS for production, but they're servers. FAISS is a file. For nine indexes that fit in memory, a server is overhead.

**Pinecone.** Cloud-hosted. APG data must not leave your machine for a PoC. Disqualified.

**Voyage AI / Cohere embeddings.** Higher-quality embeddings, but paid and remote. BGE-small-en is good enough at this scale and stays local.

**Llama / Ollama / local LLM.** Requires hardware you don't have, would tank the demo. Anthropic API is fine.

**LangGraph.** Stateful agent graphs. Genuinely useful for complex multi-step agents. Overkill for this scope and a different mental model than what you need to learn first.

**FastAPI + frontend split.** Production architecture. Skip for demo. Streamlit is the right MVP shape.

**Docker.** Production concern. If Ryan greenlights the project, this is week 1 of v2.

**LCEL (LangChain Expression Language) `|` syntax.** We use traditional `Runnable` classes and `Chain` patterns instead. LCEL is elegant once you know it but adds a vocabulary on top of Python that's confusing while you're still learning the underlying concepts. We'll use plain `RunnablePassthrough`, `RunnableLambda`, and explicit chain composition.

### Versions to pin

After you install, pin exact versions in `requirements.txt` so the build is reproducible. Use `pip freeze > requirements.lock` after a successful install and commit both files. Do not blindly copy version numbers from this document — package versions move; install fresh and pin what worked.

The package list (without versions, you pin them yourself):

```
langchain
langchain-community
langchain-anthropic
langchain-huggingface
sentence-transformers
faiss-cpu
rank-bm25
streamlit
python-dotenv
anthropic
pydantic
```

---

## 4. Architecture — three views

Architecture is hard to communicate with one diagram because it has three orthogonal aspects. We'll build three diagrams (Section 13) and refer to them throughout. Here are the textual versions.

### 4A. Layer view — what runs where

```
┌──────────────────────────────────────────────────────────────┐
│  USER (browser)                                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼─────────────────────────────────────┐
│  STREAMLIT APP (your laptop, localhost:8501)                 │
│  • Chat UI, session state, escalation counter                │
└────────────────────────┬─────────────────────────────────────┘
                         │ Python function calls
┌────────────────────────▼─────────────────────────────────────┐
│  LANGCHAIN PIPELINE (your laptop, in-process)                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  EnsembleRetriever  (BM25 + FAISS in parallel)       │    │
│  │  → CrossEncoderReranker (top-k filter)               │    │
│  │  → PromptTemplate (system + history + chunks + query)│    │
│  │  → ChatAnthropic  (with tools, caching, streaming)   │    │
│  └──────────────────────────────────────────────────────┘    │
└────────┬──────────────────────────────────────┬──────────────┘
         │ HF model (in-memory)                 │ HTTPS
┌────────▼─────────────┐         ┌──────────────▼──────────────┐
│  EMBEDDING + RERANK  │         │  ANTHROPIC API              │
│  (BGE-small-en,      │         │  Claude Haiku 4.5           │
│   ms-marco-MiniLM)   │         │  Tool calls + streaming     │
│  Local, free         │         │  Pay per token              │
└──────────────────────┘         └─────────────────────────────┘

Storage (local files):
  /kb        — markdown KB files
  /indexes   — FAISS indexes per topic
  /logs      — JSON query logs
```

The whole system except the LLM call lives on your laptop. That's the security story for Ryan: no APG content leaves your machine except the user's question and the retrieved chunks, which go to Anthropic. Even those don't go to Microsoft, OpenAI, or any party other than the LLM vendor.

### 4B. Data flow — one user question, end to end

```
User types: "Does my parcel to Saudi Arabia need duties?"
   │
   ▼
[1] Streamlit captures input, appends to chat history
   │
   ▼
[2] Pipeline receives (question, chat_history)
   │
   ▼
[3] EnsembleRetriever runs BM25 + FAISS over ALL 9 indexes
       BM25 hits "Saudi Arabia", "SAR" exactly
       FAISS hits semantically similar items (thresholds, duties)
   │   → returns top 20 candidate chunks
   ▼
[4] CrossEncoderReranker re-scores the 20 candidates
       against the original question
   │   → returns top 5 by relevance
   ▼
[5] PromptTemplate assembles:
       [System: APG persona + hard rules + tool instructions]
       [Cached prefix marker]
       [Retrieved chunks]
       [Conversation history]
       [User question]
   │
   ▼
[6] ChatAnthropic streams the response from Claude Haiku 4.5
       If the model decides to call a tool:
         → tool dispatcher runs the function
         → result fed back to model
         → model continues generation
   │
   ▼
[7] Streaming tokens render in Streamlit as they arrive
   │
   ▼
[8] Final answer + metadata logged to /logs/queries.jsonl
   │
   ▼
[9] "Did this answer your question?" feedback prompt shown
       Yes → reset escalation counter
       No  → increment; at 3, show escalation message
```

### 4C. Component view — what's in each module

This is the file structure (Section 6) viewed as a dependency graph. Modules higher in the diagram depend on those below.

```
                          app.py (Streamlit UI)
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
           chain.py                        memory.py
        (orchestration)                  (chat history)
                │
        ┌───────┼────────┬─────────┬──────────┐
        ▼       ▼        ▼         ▼          ▼
   retriever  llm.py  prompts  tools.py   logging
      │         │      .py        │          │
      │         │                 │          │
      ▼         │                 ▼          │
  embedder      │            (mock APIs)     │
      │         │                            │
      ▼         ▼                            ▼
   loader   anthropic SDK                  /logs/
      │
      ▼
   /kb files
```

---

## 5. Knowledge base strategy

You already have nine well-structured KB files in Item_NNN format. The original plan correctly identified that you do not need LangChain's character splitter — your KB is hand-chunked. We keep that decision and extend it.

### 5A. Chunking by Item boundaries

Each `## Item_NNN` heading marks the start of a chunk. The chunk runs until the next `## Item` or end of file. Each chunk gets metadata attached:

```python
{
    "page_content": "<the full text of the Item, including subheadings>",
    "metadata": {
        "source_file": "duties_thresholds_by_country.md",
        "topic": "thresholds",
        "item_id": "Item_004",
        "item_title": "Duty-Free Thresholds by Country — Reference Table"
    }
}
```

Why metadata matters: when the reranker returns the top 5 chunks, you can show in your demo logs *which Item from which file answered each question*. That's the auditability story Ryan will care about — Copilot Studio cannot do this cleanly.

### 5B. One index per topic, but searched in parallel

The original plan built one FAISS index per file (good) and routed each query to one index (bad). We keep the per-file indexing but search **all of them at once** and let reranking decide what's relevant.

Why this is better:

- **Robustness to ambiguous queries.** "I can't pay" might hit `how_to_pay` and `refusing_payment`. With a router, you bet on one. With ensemble + rerank, you get both candidates and the reranker picks the right one based on the actual question text.
- **No router model to maintain.** No extra LLM call, no extra cost, no extra failure mode, no separate eval needed for the router.
- **Works for cross-topic questions.** The exact issue Ryan flagged in your call ("my parcel is held and I can't use PayPal" — that's two topics) becomes trivial: both Items get retrieved, both get reranked highly, both go into the prompt.

Performance-wise: searching 9 small indexes is essentially free. Each index has at most ~10–15 Items. We're talking microseconds.

### 5C. The `duties_common.md` file

This file contains shared facts (APG role in duties, contact details). The original plan said "always search common as fallback". We do something simpler: it just becomes one of the 9 indexes searched in parallel. If it's relevant, the reranker brings it in. If not, it doesn't pollute the prompt.

### 5D. Future-proofing: how to add a topic later

When Ryan adds a new topic (e.g. "Collecting a parcel"):

1. Drop the new markdown file in `/kb/`.
2. Run `python -m src.embedder` to rebuild indexes.
3. Add the new file path to the EnsembleRetriever config (one line).
4. No router to retrain. No prompts to change. No regression risk for existing topics.

This is the operational story you sell Ryan. Topic addition is now a 5-minute job, not a Copilot Studio rebuild.

---

# Part II — The Build

## 6. Project structure (annotated)

This is the final state. Build it incrementally as the sessions instruct. Do not pre-create files you don't need yet — empty files are technical debt.

```
apg-rag-demo/
│
├── .env                       Session 0   API keys. NEVER commit.
├── .gitignore                 Session 0   Excludes .env, /indexes/, /logs/, venv
├── README.md                  Session 0   Public-facing description (no APG IP)
├── requirements.txt           Session 0   Pinned package versions
├── requirements.lock          Session 0   Output of pip freeze (full tree)
│
├── kb/                        Session 0   APG markdown files. NEVER pushed to GitHub.
│   ├── duties_common.md
│   ├── duties_what_are_duties_and_taxes.md
│   ├── duties_thresholds_by_country.md
│   ├── duties_order_held_pending_payment.md
│   ├── duties_how_to_pay.md
│   ├── duties_id_verification.md
│   ├── duties_refusing_payment.md
│   ├── wismo.md
│   └── damaged_goods.md
│
├── indexes/                   Session 2   Generated FAISS files. NOT committed.
│   ├── thresholds/
│   ├── order_held/
│   └── ... (one per KB file)
│
├── src/
│   ├── __init__.py
│   ├── config.py              Session 1   Constants: model names, paths, k values
│   ├── loader.py              Session 1   KB → list of Documents with metadata
│   ├── embedder.py            Session 2   Documents → FAISS index, saved to disk
│   ├── retriever.py           Session 5   EnsembleRetriever + Reranker
│   ├── llm.py                 Session 3   ChatAnthropic with caching + streaming
│   ├── prompts.py             Session 4   System prompts, hard rules, persona
│   ├── tools.py               Session 8   get_tracking_status, escalate_to_human
│   ├── memory.py              Session 9   Chat history wrapper
│   ├── chain.py               Session 4   The full LangChain pipeline (extended later)
│   └── logger.py              Session 6   JSON structured logging
│
├── app.py                     Session 10  Streamlit entry point
│
├── eval/
│   ├── questions.json         Session 11  33 eval questions + ground truth notes
│   ├── judge_prompts.py       Session 11  LLM-as-judge rubric
│   ├── run_eval.py            Session 11  Runs all questions, calls judge, scores
│   ├── compare.py             Session 12  Diff two eval runs (e.g. v1 vs v2)
│   └── results/               Session 11  CSV outputs per run, timestamped
│
├── logs/                      Generated   /queries.jsonl — one line per query
│
├── diagrams/                  Phase 0+    draw.io files (XML, diff in git)
│   ├── 01_layer_view.drawio              Phase 0 — where things run
│   ├── 02_data_flow.drawio               Phase 0 — one query end-to-end
│   ├── 03_component_view.drawio          Phase 0 → S14 — code structure
│   ├── 04_tool_sequence.drawio           Session 8 — tool-call flow (Mermaid)
│   ├── 05_feedback_state.drawio          Session 10 — escalation state machine
│   ├── 06_comparison.drawio              Session 14 — Copilot Studio vs Local RAG
│   ├── sources/                          Mermaid / mxGraph source snippets
│   └── exports/                          PNG exports for the demo deck
│
└── scripts/                   As needed   One-off utilities
    ├── reset_indexes.sh
    └── tail_logs.sh
```

### What goes to GitHub vs. what stays local

| Path | GitHub? | Reason |
|---|---|---|
| `/src/`, `/eval/`, `app.py`, `requirements.txt`, `README.md`, `/diagrams/` | Yes | Your code, your IP |
| `.env` | No (gitignored) | Contains API key |
| `/kb/` | No (gitignored) | APG content, not yours to publish |
| `/indexes/` | No (gitignored) | Generated artefact, not source |
| `/logs/` | No (gitignored) | May contain user PII in real usage |

This separation is part of what you tell Ryan. The repo is showable to anyone; the company data never leaves your laptop.

---

## 7. Phase 0 — Planning checkpoint

You have already done most of Phase 0 by reading this document. Phase 0 ends when **all** of the following are true. Do not start Phase 1 until they are.

You have installed Python 3.11+, VS Code, and Git. Verified with `python --version`, `git --version`, `code --version` from your terminal.

You have created the empty project folder, the virtual environment, and activated it. Your shell prompt shows `(venv)` when you cd into the project. You can pip install. You have not yet installed any packages — that's Session 1.

You have your Anthropic API key in `.env`. You have a hard spend limit set on the Anthropic console (e.g. $20/month, well above what you'll need but caps blast radius if something loops).

You have built **the three Phase 0 diagrams** (1, 2, and a draft of 3) following the specs and workflow in Section 13. They do not need to be pretty — they need to be accurate. You can walk through Diagram 2 (data flow) from memory on a whiteboard. This is the gating checkpoint. Diagrams 4, 5, 6 are built later (Sessions 8, 10, 14) when the underlying code exists.

You have a private GitHub repo created, with `.gitignore` already pushed. Your initial commit message is `"chore: project scaffold"` — see Section 20 for commit conventions.

You have the KB files copied into `/kb/`. You have spot-checked one of them in VS Code and confirmed the Item structure is intact.

If any of these isn't true, stop and finish it. The two hours you save by skipping this are the ten hours you'll lose later when something breaks and you can't tell whether it's an environment issue or a code bug.

---

## 8. Phase 1 — Core RAG pipeline (Days 1–4)

Goal of Phase 1: a question goes in, a grounded answer comes out, end to end. Not pretty, not smart, not multi-turn. One function, one question, one answer. You prove the pipe is connected before you make the water taste good.

### Session 1 — Document loader (Day 1, ~2 hrs)

**Concept to learn.** A "Document" in LangChain is a small object with `page_content` (string) and `metadata` (dict). Every retriever, splitter, and chain in LangChain consumes lists of these. Getting your KB into Document form correctly is the foundation for everything else.

**Why this matters.** If your loader produces messy chunks (stray YAML, broken Item boundaries, duplicated content), every downstream step will silently degrade. Garbage in, garbage everywhere.

**What you build.** `src/loader.py` with this signature:

```python
from typing import List
from pathlib import Path
from langchain.schema import Document

def load_kb_file(file_path: Path) -> List[Document]:
    """
    Read a single KB markdown file and split it into Documents at Item boundaries.

    Each Document corresponds to one ## Item_NNN block. Metadata includes:
      - source_file: filename (string)
      - topic: derived from filename (e.g. "thresholds")
      - item_id: e.g. "Item_004"
      - item_title: text of the heading after the colon

    Args:
        file_path: Path to a .md file in the kb/ folder.

    Returns:
        List of Documents, one per Item.

    Raises:
        ValueError: if no Items are found in the file (likely wrong file).
    """
    ...

def load_all_kb_files(kb_dir: Path) -> dict[str, List[Document]]:
    """
    Load every .md file in kb_dir. Return a dict mapping topic name to list of Documents.
    The topic name is derived from the filename: 'duties_thresholds_by_country.md' → 'thresholds'.
    """
    ...
```

**Manual test before moving on.** Run a tiny script that calls `load_kb_file("kb/duties_thresholds_by_country.md")` and prints the result. You should see four or five Document objects, each with a single Item's content and correct metadata. Verify item titles match what's in the source file.

**Checkpoint.** You can describe in plain English what `load_all_kb_files` returns. You can point at any chunk and tell me which file and which Item it came from from the metadata alone.

**Common pitfall.** Markdown content sometimes has `##` inside Item bodies (subheadings). Make sure your splitter only splits on `## Item_` specifically, not any `##`.

### Session 2 — Embedding and FAISS indexes (Day 1, ~2 hrs)

**Concept to learn.** An embedding model takes a string and returns a fixed-length vector of floats (BGE-small returns 384 floats). Two strings that mean similar things produce vectors that are near each other in 384-dimensional space ("near" measured by cosine similarity). FAISS is a library that, given thousands of these vectors, can find the closest ones to a query vector very quickly.

**Why BGE-small over MiniLM.** MiniLM-L6-v2 is fine. BGE-small-en-v1.5 is meaningfully better on retrieval benchmarks (about 5 percentage points higher MTEB retrieval score) at almost identical size and inference speed. The migration cost is one model name string. Take the upgrade.

**What you build.** `src/embedder.py`:

```python
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def get_embedder() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.
    On first call, downloads the model (~130 MB) to ~/.cache/huggingface.
    Subsequent calls are instant.
    """
    ...

def build_index_for_topic(
    documents: List[Document],
    topic_name: str,
    output_dir: Path
) -> None:
    """
    Embed the documents and persist a FAISS index to output_dir / topic_name /.
    Overwrites if the index already exists.
    """
    ...

def build_all_indexes(kb_dir: Path, indexes_dir: Path) -> None:
    """
    Top-level function. Loads all KB files, builds one FAISS index per topic, saves to disk.
    Logs the count of vectors per index.
    """
    ...
```

**Manual test.** Run `python -m src.embedder` from project root. Confirm `/indexes/` fills up with one subfolder per topic, each containing `index.faiss` and `index.pkl`. Open a Python shell, load one index, run `index.similarity_search("how long is the holding period", k=3)`, eyeball the results — they should be from `order_held`.

**Checkpoint.** You can explain why we save indexes to disk instead of rebuilding on startup (rebuilding is slow, indexes are static until KB changes). You can recreate the indexes from scratch if `/indexes/` is deleted (one command).

### Session 3 — Anthropic API call (Day 2, ~2 hrs)

**Concept to learn.** A "completion" or "message" is a single API call to Anthropic's `/v1/messages` endpoint. You send a system prompt (instructions), a list of messages (conversation), and parameters (model, max_tokens, temperature). You get back a response object with `.content` (the text) plus token counts and stop reason.

**Why temperature matters here.** For RAG-grounded Q&A, you want temperature low (0.0–0.2). High temperature is for creative generation; here it just adds variance and makes evals noisy.

**What you build.** `src/llm.py`:

```python
from langchain_anthropic import ChatAnthropic

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

def get_llm(streaming: bool = False) -> ChatAnthropic:
    """
    Return a configured ChatAnthropic instance.
    Reads ANTHROPIC_API_KEY from env.
    Default temperature 0.0 for deterministic answers.
    Default max_tokens 1024 (RAG answers should be short).

    streaming=True for Streamlit UI; False for eval runs and tests.
    """
    ...
```

**Manual test.** Get the model to answer a trivial question without any RAG: `llm.invoke("What's 2+2? One word answer.")`. Confirm you get "4" (or "Four"). This proves the API key works and the SDK is wired up.

**Checkpoint.** You can explain what a token is, roughly how many tokens fit in 1000 characters of English (~250), and where to find your spend on the Anthropic console.

### Session 4 — First end-to-end chain (Day 2 or Day 3, ~2 hrs)

**Concept to learn.** A "chain" in LangChain is a sequence of steps where each step's output feeds the next step's input. The simplest possible RAG chain: take a question → retrieve chunks → format prompt → call LLM → return answer. We'll build the **simple** version first (pure FAISS, single topic, no rerank, no memory). Phase 2 makes it sophisticated.

**Why this stage matters.** This is your "first light" moment. Once a question goes in and a real answer comes out, every subsequent change is incremental. Before this, everything is theoretical.

**What you build.** `src/prompts.py` (initial version) and `src/chain.py`:

```python
# prompts.py
RAG_SYSTEM_PROMPT_V1 = """You are APG Chat, the customer support assistant for APG eCommerce.

Answer the user's question using ONLY the context provided below. Do not use prior knowledge.
If the context does not contain enough information to answer, say so and direct the user to
APG support: generalsupport@apgecommerce.com.

Tone: professional, friendly, concise. No hedging. No "I think" or "might".

Context:
{context}
"""

# chain.py
def build_simple_chain(topic: str):
    """
    Build a minimal RAG chain over a single topic's FAISS index.
    No memory, no rerank, no streaming. Just the pipe.

    Returns a Runnable that accepts {"question": str} and produces a string answer.
    """
    ...

def ask_simple(question: str, topic: str) -> str:
    """Convenience wrapper for testing from the CLI."""
    ...
```

**Manual test.** From terminal:
```
python -c "from src.chain import ask_simple; print(ask_simple('How long is the holding period for ASOS orders to Israel?', 'order_held'))"
```
You should get an answer mentioning 14 days. Without that, do not proceed.

**Checkpoint.** You can run `ask_simple` against three different topics with three different questions and get sensible answers. You understand what's in the prompt being sent to Claude (you should add a `print(prompt)` line temporarily to see it, then remove it).

**End of Phase 1.** Pipe is connected. Bad answers? Fine. Slow? Fine. Hard-coded topic? Fine. The point of Phase 1 was first light, and you have it.

---

## 9. Phase 2 — Hybrid retrieval and reranking (Days 5–7)

Goal: replace the toy retriever from Phase 1 with the production-shape retriever. This is where accuracy comes from.

### Session 5 — EnsembleRetriever (Day 5, ~3 hrs)

**Concept to learn.** Two retrievers running in parallel, results merged. BM25 (keyword) catches exact matches the embedder fumbles. Dense retrieval (FAISS) catches paraphrases the keyword search misses. Together they cover both modes.

**Why hybrid retrieval is the single biggest accuracy win.** Test case: "What's the threshold for KSA?" The embedder may not know "KSA" maps to Saudi Arabia (depends on training data). BM25 will retrieve any Item containing "KSA" if you've added that as an alias. More realistic: "What about my parcel from ASOS?" Pure semantic search may surface generic ASOS-shaped content; BM25 finds the literal token "ASOS" in the Items where it actually appears.

**What you build.** `src/retriever.py`:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

def build_retriever(
    indexes_dir: Path,
    kb_dir: Path,
    bm25_weight: float = 0.4,
    faiss_weight: float = 0.6,
    k_per_retriever: int = 10
) -> EnsembleRetriever:
    """
    Construct an EnsembleRetriever that searches all topic indexes simultaneously.

    Internally:
      - One BM25Retriever built from all KB Documents (in-memory, fast)
      - Multiple FAISS retrievers (one per topic) wrapped into a single dense retriever
      - EnsembleRetriever combines results with weighted reciprocal rank fusion

    Returns top (k_per_retriever * 2) candidates by default.

    Tune bm25_weight and faiss_weight in eval. Default 0.4/0.6 favours semantic
    slightly because most queries are paraphrases, not exact-match.
    """
    ...
```

**Manual test.** Run the same three Phase 1 questions through this retriever (without LLM yet). Print the retrieved chunks. Confirm:
- The Saudi Arabia question retrieves the Saudi-specific row, not generic threshold content.
- The PayPal question retrieves the payment-methods Item.
- The "where is my parcel" question retrieves wismo Items.

**Checkpoint.** You can explain what reciprocal rank fusion does in one sentence. You can describe one query for which you'd expect BM25 to dominate and one where FAISS would.

### Session 6 — Cross-encoder reranking (Day 5 or 6, ~2 hrs)

**Concept to learn.** A cross-encoder takes (query, candidate) as a pair and outputs a single relevance score. Unlike the embedder (which scores query and candidate independently and compares vectors), the cross-encoder lets the model attend to both texts together — much more accurate for relevance, but too slow to run on the whole corpus. The pattern is: retrieve cheaply (bi-encoder + BM25), rerank expensively (cross-encoder) on the top 20 → top 5.

**Why this earns its keep.** Empirically, reranking is worth more than almost any prompt tweak. On RAG benchmarks the lift is often 5–15 percentage points in answer quality. The cost is ~50ms of local inference. Free.

**What you build.** Extend `src/retriever.py`:

```python
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RerankingRetriever:
    """
    Wraps an EnsembleRetriever. Retrieves k_initial candidates, reranks with
    a cross-encoder, returns top k_final.
    """
    def __init__(self, base_retriever, k_initial: int = 20, k_final: int = 5):
        ...

    def get_relevant_documents(self, query: str) -> List[Document]:
        ...
```

**Manual test.** Same three queries as before. Compare reranked output vs un-reranked. The reranked top-5 should look noticeably tighter — fewer false positives.

**Checkpoint.** You can articulate why a cross-encoder is more accurate but slower than a bi-encoder. You understand why we run it only on the top-20 candidates rather than the whole corpus.

### Session 7 — Plug retriever into chain, add structured logging (Day 6 or 7, ~3 hrs)

**Concept to learn.** A chain is just data flowing through transformations. We swap the simple retriever for the reranking retriever, and at the same time add structured logging so we can see what was retrieved for every query.

**What you build.** Update `src/chain.py` to use the reranking retriever; create `src/logger.py`:

```python
def log_query(
    question: str,
    retrieved_chunks: List[Document],
    final_answer: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    tools_called: list[str] | None = None
) -> None:
    """
    Append a JSON line to logs/queries.jsonl with full trace of one query.
    Used for debugging, eval baselines, and the demo's "look how observable this is" moment.
    """
    ...
```

**Manual test.** Ask three questions. `tail -f logs/queries.jsonl` in another terminal — you should see one well-formatted JSON object per query, parseable with `jq`.

**Checkpoint.** You can demonstrate the observability story to me right now: open a log file, point at a query, show the retrieved chunks alongside the answer, and explain whether the answer was grounded in those chunks.

**End of Phase 2.** Retrieval is solid. Answers are grounded. You can debug bad answers by reading the log.

---

## 10. Phase 3 — Tools, memory, streaming (Days 8–10)

Goal: replicate the Copilot Studio bot's interactive features. Tracking lookup, escalation, multi-turn memory, and streaming UX.

### Session 8 — Tool definitions (Day 8, ~3 hrs)

**Concept to learn.** Anthropic's tool use lets Claude decide *during generation* that it needs to call an external function. You define tools as JSON schemas with name, description, and input shape. Claude returns a `tool_use` block instead of (or alongside) text. Your code dispatches the call, runs the function, and feeds the result back to Claude in a follow-up turn.

**Why native tool use, not an "agent" framework.** Agent frameworks (LangChain Agents, CrewAI) wrap tool use in extra abstraction. You don't need that — you have two tools, both deterministic. Native tool calling via `bind_tools()` on `ChatAnthropic` is exactly what you want and one layer thinner.

**What you build.** `src/tools.py`:

```python
from langchain_core.tools import tool

@tool
def get_tracking_status(tracking_number: str) -> dict:
    """
    Look up the current status of a parcel by tracking number.

    For the demo, returns mocked data based on a simple lookup table that
    covers the demo scenarios:
      - 'APG12345' → "In transit, expected Tuesday"
      - 'APG99999' → "Held pending duties payment"
      - anything else → "Tracking number not found"

    In production this would call the APG tracking API.
    """
    ...

@tool
def escalate_to_human(reason: str, conversation_summary: str) -> dict:
    """
    Create an escalation ticket and return a ticket reference.

    For the demo: appends a record to logs/escalations.jsonl and returns
    a synthetic ticket ID (e.g. 'TKT-20260505-0001').

    In production this would call APG's ticketing system via Power Automate.
    """
    ...
```

You then bind them to the LLM:

```python
llm_with_tools = get_llm().bind_tools([get_tracking_status, escalate_to_human])
```

And handle the tool-call loop in the chain.

**Manual test.** Ask "Where is my parcel APG99999?" — the model should call `get_tracking_status`, get back the held-for-duties response, and produce a final answer that mentions duties and references the relevant KB content. This is the killer demo moment: tool use + RAG composing in one turn.

**Checkpoint.** You can explain the difference between a "tool call" turn and a "final answer" turn from the model. You can read a log entry and tell me whether tools were used.

### Session 9 — Conversation memory (Day 9, ~2 hrs)

**Concept to learn.** LangChain's `RunnableWithMessageHistory` wraps any chain to give it per-session message history. You provide a session ID and a function that returns the history for that ID; LangChain handles injecting it into the prompt and updating it after each turn.

**Why this matters for your demo.** Ryan's call transcript flagged exactly this issue: the Copilot Studio bot doesn't handle topic switches gracefully. With proper conversation memory + the hybrid retriever + the right system prompt, this becomes natural. User asks about Israel duties, then says "what about Saudi?" — model reads history, knows the topic is duties, retrieves Saudi rows, answers correctly.

**What you build.** `src/memory.py`:

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-memory session store. For a real deployment you'd use Redis or similar.
_sessions: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Return the message history for a session, creating it if new."""
    ...

def wrap_chain_with_memory(chain):
    """Return a chain that automatically reads/writes session history."""
    ...
```

**Manual test.** Three-turn conversation:
1. "Do I need to pay duties on my parcel to Israel?"
2. "How long do I have to pay?"
3. "What if I just don't?"

Confirm turn 2 retrieves Israel-specific holding period (14 days for ASOS). Confirm turn 3 covers the refusing-payment topic.

**Checkpoint.** You can explain what the prompt looks like on turn 3 (system + previous 2 user messages + previous 2 assistant messages + retrieved chunks for turn 3 + new user question).

### Session 10 — Streaming + Streamlit UI (Day 10, ~3 hrs)

**Concept to learn.** Streaming returns tokens as they're generated rather than waiting for the full response. In LangChain you call `.stream()` instead of `.invoke()` and iterate over the chunks. In Streamlit, `st.chat_message` + `st.write_stream` renders incrementally — exactly what you need for a chat UI that feels alive.

**Prompt caching note.** When you make the API call, mark the system prompt and retrieved chunks with `cache_control={"type": "ephemeral"}`. This caches them for ~5 minutes. Subsequent queries with the same system prompt (which is most of them) hit the cache for ~90% discount. Configure this on `ChatAnthropic` via `extra_headers` or the `cache` parameter (check current SDK docs at install time — the API surface evolves).

**What you build.** `app.py`:

```python
import streamlit as st
from src.chain import build_full_chain
from src.memory import wrap_chain_with_memory

# Session state holds: messages list, escalation counter, session_id

def render_chat():
    """Display chat history, render new user input, stream assistant response."""
    ...

def render_feedback():
    """Did this answer your question? Yes / No buttons. Manage escalation counter."""
    ...

def main():
    st.set_page_config(page_title="APG Chat (local)", page_icon="📦")
    # ... compose render_chat + render_feedback + sidebar with session controls
```

**Manual test.** Run `streamlit run app.py`. Have a 5-turn conversation. Observe streaming. Test the feedback loop — three "No" responses should trigger an escalation message and either offer the escalate tool or directly call it.

**Checkpoint.** The UI feels responsive. You can explain what's in `st.session_state` at any moment. You can pull up the log file and show what got retrieved for each turn.

**End of Phase 3.** The bot is feature-complete with the current Copilot Studio capabilities and exceeds them on observability.

---

## 11. Phase 4 — Eval, polish, demo prep (Days 11–14)

Goal: prove it works, prepare the cost case, and rehearse.

### Session 11 — Evaluation harness (Day 11, ~3 hrs)

**Concept to learn.** "LLM-as-judge" means you use a strong language model with a fixed rubric to grade your bot's answers against ground truth. It's not perfect, but it's reproducible, fast, and at this scale (33 questions) it's the right tool.

**Rubric we'll use.** Three dimensions, each scored 0–2:
- **Factual correctness** — does it match the KB ground truth?
- **Groundedness** — is every claim traceable to retrieved chunks (no hallucination)?
- **Tone** — does it match the APG persona (concise, no hedging, friendly-authoritative)?

Total score per question: 0–6. Pass threshold: ≥5. Target: 27/33 questions passing (matches the original plan's 80% bar).

**What you build.** `eval/judge_prompts.py` (the rubric prompt), `eval/run_eval.py`:

```python
def run_eval(questions_file: Path, output_dir: Path, run_label: str) -> dict:
    """
    For each question in questions_file:
      1. Run it through the full pipeline (no memory; each question is fresh)
      2. Capture answer + retrieved chunk metadata + token counts
      3. Send (question, ground_truth, answer) to judge LLM
      4. Parse scores
    Write results CSV to output_dir/run_label.csv.
    Return summary stats: pass rate, mean score per dimension, total cost.
    """
    ...
```

**Manual test.** Run the eval. Inspect results CSV in Excel. Find the worst-performing questions. Read their logs. Diagnose: was it bad retrieval or bad generation?

**Checkpoint.** You have a numerical baseline. You can re-run the eval after any change and produce a comparable number.

### Session 12 — Tuning round (Day 12, ~3 hrs)

Now you have an eval, you can do iterative improvement. Prioritise in order:

First, look at the failed questions. For each, classify the failure:
- Retrieval failed (right Item wasn't in the top 5) → adjust BM25/FAISS weights, rerank k, or KB phrasing
- Generation failed (right Item retrieved, wrong answer produced) → improve system prompt
- Tool routing failed (should have called a tool, didn't) → tool descriptions

Then re-run the eval. Compare before/after. Commit the change with a message like `"feat(retriever): bump bm25 weight to 0.5, +3 questions passing"`.

Stop when you hit your 27/33 target or when one full session of tuning produces zero improvement (whichever comes first).

**Checkpoint.** Eval target hit, or you've documented why it isn't and what's needed.

### Session 13 — Cost model and comparison (Day 13, ~2 hrs)

**What you build.** `eval/cost_report.py` and a one-page markdown summary.

The numbers to produce:

| Metric | Copilot Studio | Local RAG (Haiku 4.5) |
|---|---|---|
| Pricing model | Flat per message bundle | Pay per token |
| Effective cost per session | (your existing figure) | (computed from eval avg tokens) |
| Cost per 1,000 sessions | | |
| Cost at current monthly volume | | |
| Cost at 2x monthly volume | | |
| Cost with prompt caching enabled | n/a | (with ~70% cache hit rate assumption) |

The story: Copilot Studio's cost scales linearly with usage and is tied to a 25k cap. Local RAG cost scales with tokens, dominated by output (input is mostly cached). At your projected volumes, we save X. At 2x volumes, we save Y.

Important: be honest about what the comparison excludes. Microsoft licence is a sunk cost (probably). Engineering time to build/maintain RAG is real. Hosting cost in production (a small VM) is real but small.

**Checkpoint.** You have a single-page comparison Ryan can show to Paulina without further work.

### Session 14 — Diagrams polish, demo rehearsal, README (Day 14, ~3 hrs)

Update all six draw.io diagrams to match what you actually built (the data flow and component view will have drifted; the comparison diagram is built fresh in this session). Use the Mermaid → draw.io workflow from Section 13A — do not draw boxes by hand. Export all six to PNG with transparent background, 2x scale, into `/diagrams/exports/`.

Write `README.md` for the public repo. It should explain: what this is, what it isn't, the architecture in 3 paragraphs, how to run it, link to the diagrams. No APG-specific facts. No KB content. No company secrets.

Rehearse the demo (Section 15) with a colleague or alone with a stopwatch. Time it. The first run will be too long. Cut it.

Record a 5-minute screen capture as a fallback in case the live demo can't happen (Ryan's network, time pressure, etc.). Use OBS or Loom.

**Checkpoint.** You've delivered the demo to yourself, on the clock, and stayed within 5 minutes. You have a recorded backup.

---

# Part III — The Demo

## 12. Evaluation framework (detail)

The eval is what gives your demo credibility. "I ran 33 questions and it got 28 right" is a much stronger claim than "it works".

### Structure of `eval/questions.json`

```json
[
  {
    "id": "DUT-001",
    "question": "Does my order to Saudi Arabia need duties?",
    "expected_topic": "thresholds",
    "ground_truth_summary": "Saudi Arabia threshold is SAR 1,000 inclusive of cost, insurance, and freight. Below that, no duties. At or above, duties apply, 7-day holding period.",
    "must_mention": ["SAR 1,000", "CIF or cost insurance freight", "7 days"],
    "must_not_mention": ["PayPal", "USD"]
  },
  ...
]
```

The `must_mention` / `must_not_mention` arrays let the judge do quick deterministic checks alongside the qualitative score.

### What the judge sees

The judge LLM (also Claude, but a separate call with no shared state) is given:
- The question
- The ground truth summary
- The bot's actual answer
- The rubric

It returns a JSON object with the three scores and a one-line justification per dimension. You parse this in `run_eval.py`.

Why use Claude as the judge when Claude generated the answer? Because the judge has the *ground truth* in front of it, and the bot didn't. The asymmetry is what makes it valid. (If you have appetite later, swap to a different model for the judge to remove correlation.)

### Comparing against Copilot Studio

The honest comparison is: run the same 33 questions through both bots, score both with the same judge, compare. You probably already have a baseline from your existing APG_Eval_Baseline_v1 work in Copilot Studio (per your project context). If yes, reuse those scores. If no, run the questions through Copilot Studio yourself and score manually using the same rubric, then re-score automatically when you can.

---

## 13. draw.io — workflow and diagram catalogue

This is the most leveraged tool in your kit and the original plan barely used it. Six diagrams, not three. Built in roughly 20 minutes each (not two hours each) using the workflow below.

### 13A. The workflow that makes diagrams cheap

**The mistake people make with draw.io.** They open it and start drawing boxes. For a six-diagram project that's two days of dragging rectangles. Don't.

**The real workflow.** draw.io desktop app supports text-based diagram input via `Extras → Edit Diagram…`. The dialog accepts two formats: **mxGraph XML** (draw.io's native format) and **Mermaid** (a much simpler text language). Both render straight into a working diagram.

Mermaid is the killer because it's compact, readable, and Claude generates it perfectly. For sequence diagrams, state machines, flowcharts, and class-style component graphs, Mermaid → draw.io takes three minutes. mxGraph XML is the right tool for diagrams with custom layouts and exact positioning (like the layer view) where Mermaid's auto-layout doesn't give you what you want.

**The end-to-end loop.**

1. Decide which diagram you need and which format suits it (Mermaid for flow/sequence/state, mxGraph XML for fixed-layout).
2. Ask Claude (in chat — this is exactly the kind of task to hand off) to generate the source. Give it the bible's relevant section as context plus a one-line description of what you want.
3. In draw.io: `File → New → Blank diagram`. Then `Extras → Edit Diagram…`. Paste the source. Click OK.
4. Apply the colour palette manually (Mermaid won't enforce it; mxGraph XML can if Claude was told to use it). Five minutes.
5. Save the `.drawio` file in `/diagrams/`. These are XML under the hood, so they diff cleanly in git — commit them like code.
6. When you're ready for the demo deck, `File → Export As → PNG` with transparent background and 2x scale. Save under `/diagrams/exports/`.

**Prompts that actually work for step 2.**

For sequence diagrams:
> "Generate a Mermaid sequenceDiagram for the tool-call flow in PROJECT_BIBLE.md Section 4B step 7. Actors: User, Streamlit UI, LangChain Pipeline, Claude API, get_tracking_status tool. Scenario: user asks 'Where is APG99999?', the model issues a tool_use block, the dispatcher runs the tool, the result is fed back as a tool_result message, and the model produces a final answer that grounds in retrieved KB chunks about held-for-duties parcels."

For state machines:
> "Generate a Mermaid stateDiagram-v2 for the feedback / escalation logic. States: AwaitingQuestion, Generating, AwaitingFeedback, Escalated. Transitions: question_received, answer_streamed, feedback_yes (resets counter, returns to AwaitingQuestion), feedback_no (increments counter, if counter ≥ 3 transitions to Escalated, else AwaitingQuestion). Note that 'Escalated' offers `escalate_to_human` and ends the session."

For component graphs:
> "Generate a Mermaid graph TD diagram of the modules in PROJECT_BIBLE.md Section 6. Top: app.py. Second row: chain.py (and a separate node memory.py). Third row, dependencies of chain.py: retriever.py, llm.py, prompts.py, tools.py, logger.py. Fourth row: embedder.py and loader.py, plus the /kb and /indexes data nodes. Annotate each module with the session number that built it."

For mxGraph XML (fixed-layout layer or comparison diagrams):
> "Generate mxGraph XML for an A4-landscape draw.io diagram showing two side-by-side architectures: 'Copilot Studio (current)' on the left, 'Local RAG (proposed)' on the right. Use the colour palette in PROJECT_BIBLE.md Section 13C: green #C8E6C9 for local components, orange #FFE0B2 for external APIs, grey #E0E0E0 for storage. Each side should have 5–6 boxes stacked vertically with arrows. Title at top, legend at bottom-right."

**Why this is leverage.** A sequence diagram drawn by hand in draw.io takes 30–45 minutes for someone fluent. The Mermaid version takes one minute to describe, ten seconds to generate, two minutes to paste-and-tweak. Same output, 20x faster. You spend the saved time on the diagrams that actually need hand-tuning (the layer view, the comparison diagram).

### 13B. The diagram catalogue

Six diagrams. Three are the originals (renumbered and tightened). Three are new and have higher demo impact than the originals. Build all six during Phase 0 and Session 14, in the order below.

| # | Filename | Format | Purpose | Phase |
|---|---|---|---|---|
| 1 | `01_layer_view.drawio` | mxGraph XML | Where things run; security story | Phase 0 |
| 2 | `02_data_flow.drawio` | mxGraph XML | One query end-to-end; the demo walkthrough diagram | Phase 0 |
| 3 | `03_component_view.drawio` | Mermaid graph TD | Code module dependencies | Phase 0 (light) → Session 14 (final) |
| 4 | `04_tool_sequence.drawio` | Mermaid sequenceDiagram | Tool-call flow; demo killer | Session 8, polished Session 14 |
| 5 | `05_feedback_state.drawio` | Mermaid stateDiagram-v2 | Feedback / escalation logic | Session 10 |
| 6 | `06_comparison.drawio` | mxGraph XML | Copilot Studio vs Local RAG side by side | Session 14 |

### 13C. Style guide (apply to all six)

Page size: A4 landscape (fits a slide). Font: default sans-serif, 12pt labels, 10pt annotations, 14pt bold titles. Arrows: solid 2pt, medium arrowheads. Label any arrow that represents data with the data type (`Document[]`, `string`, `List[Document]`).

The single colour palette. Use it everywhere — coherence across diagrams sells professionalism harder than any individual diagram does.

| Role | Fill | Border |
|---|---|---|
| Local processing | `#C8E6C9` | `#2E7D32` |
| External API | `#FFE0B2` | `#E65100` |
| Storage / files | `#E0E0E0` | `#424242` |
| User actions | `#BBDEFB` | `#0D47A1` |
| Decision points | `#FFF9C4` | `#F57F17` |
| Failure / kill | `#FFCDD2` | `#B71C1C` |

Mermaid doesn't apply this palette automatically — you'll need to tweak fills in draw.io after import, or use Mermaid's `classDef` blocks. Either is fine; pick what's faster.

### 13D. Per-diagram specs

**Diagram 1 — Layer view (`01_layer_view.drawio`).** Where each piece of the system runs. Used for the security narrative. Horizontal swim-lanes top to bottom: user browser (blue), your laptop (green, occupies ~60% of vertical space), Anthropic API (orange, on the right side of the laptop lane connected by HTTPS arrows), storage (grey, three boxes: `/kb`, `/indexes`, `/logs`). Inside the laptop lane: a parent box for the LangChain pipeline containing EnsembleRetriever, CrossEncoderReranker, PromptTemplate, ChatAnthropic, and the tool dispatcher; plus the embedding model and reranker model as adjacent boxes. Three floating annotations: near the laptop ("All KB content stays here. Never leaves the machine."), near the Anthropic box ("Receives only: user question + retrieved chunks + system prompt"), and near storage ("Local disk only. KB and logs gitignored.").

**Diagram 2 — Data flow (`02_data_flow.drawio`).** One question end-to-end. The diagram you walk Ryan through during the demo. Vertical flowchart, top to bottom, nine numbered boxes mapping to Section 4B steps 1–9. Critical detail: the tool branch (yellow diamond at step 7) must visibly loop back to step 6 so the multi-turn nature of tool calls is obvious. Label arrows with what flows between boxes.

**Diagram 3 — Component view (`03_component_view.drawio`).** Your codebase as a dependency tree. Generated from Mermaid graph TD. `app.py` at the top, `chain.py` and `memory.py` below it, then the dependencies of `chain.py`, then the lowest-level modules and data folders at the bottom. Each module annotated with one line: which session built it and what it does ("`embedder.py` — Session 2 — builds FAISS indexes from KB Documents").

**Diagram 4 — Tool call sequence (`04_tool_sequence.drawio`).** Mermaid sequenceDiagram. The flow when a user asks about a tracking number and the model invokes `get_tracking_status`. Five actors across the top: User, Streamlit UI, LangChain Pipeline, Claude API, Tracking Tool. The interaction shows: user message → pipeline assembles prompt → Claude returns a `tool_use` block (not a final answer) → pipeline dispatches to the tool → tool result fed back as `tool_result` message → Claude produces final answer → pipeline streams to UI → UI to user. Add an `alt` block showing "if tool returns 'not found'" to demonstrate error handling. This is the diagram that earns the "tool-using agent" credit. It is, frankly, the most impressive diagram in the deck — make it tight.

**Diagram 5 — Feedback / escalation state machine (`05_feedback_state.drawio`).** Mermaid stateDiagram-v2. Four states: `AwaitingQuestion`, `Generating`, `AwaitingFeedback`, `Escalated`. Transitions labelled with the events that trigger them: `question_received`, `answer_streamed`, `feedback_yes` (counter resets, returns to `AwaitingQuestion`), `feedback_no` (counter increments; if ≥3, transitions to `Escalated`; else returns to `AwaitingQuestion`). The `Escalated` state has one outgoing transition: `escalate_to_human` invoked, session ends.

**Diagram 6 — Comparison: Copilot Studio vs Local RAG (`06_comparison.drawio`).** mxGraph XML, two-column layout. Left column titled "Copilot Studio (current)": Microsoft cloud, GUI-defined topics, KB files in Dataverse, Power Automate flows for tracking and tickets, opaque routing logic, $200/month for 25k messages. Right column titled "Local RAG (proposed)": laptop/server, code-defined chain, KB on disk, native tool use for tracking and tickets, observable routing (link to logs), per-token cost. Arrows highlighting key contrasts: "extensibility", "observability", "cost scaling", "vendor lock-in". This is the diagram that sits next to your cost table during minute 3 of the demo — the visual sells the comparison even if Ryan only glances at the numbers.

---

## 14. Cost model

The exact numbers you'll fill in from your eval run. The shape of the comparison:

**Inputs you'll measure during eval.**
- Average input tokens per query (system + chunks + history + question)
- Average output tokens per query
- Average tools called per query
- Cache hit rate (Anthropic returns this in the response)

**Inputs you already know.**
- Copilot Studio cost per 1,000 sessions (your $200 / 25,000 messages = $8 per 1,000 messages, but be careful — sessions and messages aren't the same; use whichever Ryan reports against)
- Anthropic Haiku 4.5 pricing — look up at build time, do not trust me to remember the exact rate

**Calculation.** Per query cost ≈ (input_tokens × input_rate × cache_factor) + (output_tokens × output_rate). Cache factor: cached portion charged at ~10% of normal input rate.

**The honest framing for Ryan.**
- "At current usage, the local RAG approach costs roughly X% of Copilot Studio."
- "The savings grow with volume because Copilot Studio scales in 25k-message chunks while RAG scales linearly with tokens."
- "There's an engineering cost to maintain this — figure ~half a day per month for KB updates and bug fixes — that I'm not pricing in."
- "Power Automate flows for tracking and ticketing are not yet replicated. The tool-use scaffolding is there; the actual integrations are a couple of days of work each."

This honesty is what makes the cost case credible. If you over-claim, Ryan finds the gap and the whole pitch deflates.

---

## 15. Demo script — minute by minute

5 minutes. Stopwatch.

**Minute 0:00–0:30 — Set the frame.**
"Hey Ryan — I built a local prototype of what an APG chatbot looks like without Copilot Studio. Same KB, same topics. Runs on my laptop. The bot itself is the demo, but I'll spend most of the time on three things: how it works, how it scores against the current bot, and what it costs."

**Minute 0:30–1:30 — Architecture (Diagram 2 on screen).**
"One question, one path. The question goes through hybrid retrieval — keyword search and semantic search in parallel across all nine KB files. A cross-encoder reranker then picks the five most relevant chunks. Those chunks plus the user's question go to Claude Haiku via the Anthropic API. If the model needs to look up a tracking number or escalate, it calls a tool. The whole pipeline lives on my laptop except the LLM call. KB content never leaves the machine."

Pause for any architecture questions. Don't get pulled into a deep dive — say "happy to deep-dive on that after; let me show you it working".

**Minute 1:30–3:00 — Live demo (Streamlit on screen).**

Three queries, in this order:

1. *"Does my order to Saudi Arabia need duties?"* — exercises hybrid retrieval (proper noun + currency code). Watch the streaming. Open the log file in a side panel — point at the retrieved chunks.

2. *"My tracking shows held — APG99999."* — exercises the tracking tool. The model calls `get_tracking_status`, gets back "held pending duties payment", produces an answer that combines the tracking result with KB context about what to do next.

3. *"I want to cancel — I don't want to pay these duties."* — exercises the refusing-payment topic and the cross-topic-handling. Show that the answer is grounded and walks the user through the right process.

**Minute 3:00–4:00 — The numbers.**

Eval: "I ran the 33-question baseline through both bots. Local scored X/33. Current Copilot Studio bot scored Y/33." Show the eval CSV. Don't argue the number. Let it sit.

Cost: "On current usage, local RAG costs about A% of what Copilot Studio costs. With Anthropic prompt caching enabled, B%." Show the table.

**Minute 4:00–4:30 — What this is NOT.**
"Three things this isn't. It's not deployed — it runs on my laptop. It's not integrated with the real tracking API or the ticketing system — those tools are mocked. And it hasn't had a security review. So this isn't a launch — it's a vote on whether to invest the next phase of work."

**Minute 4:30–5:00 — The ask.**
"Two things from you. One: am I right that the cost trajectory makes this worth a serious look? Two: if yes, the next step I'd propose is a 1-week production-shape spike — Docker, FastAPI, real tracking integration. I can do that next sprint."

Then shut up. Let Ryan respond.

---

## 16. Q&A prep — what Ryan will ask

Based on his style and prior comments, expect:

**"What about cross-topic interrupts? Does this fix the issue I flagged?"**
Yes — show this live. Multi-turn conversation where you switch topics mid-flight. The hybrid retriever pulls relevant chunks regardless of which topic Copilot Studio would have routed to. Memory keeps the context.

**"What's the regression risk when we add a topic?"**
Adding a topic is: drop the markdown file in /kb/, run `embedder.py`, restart. No router to retrain, no prompts to change, no existing topics affected. Show the eval re-running on the existing topics to prove no regression.

**"Why didn't you use Copilot Studio extensions / connectors / agents?"**
Two reasons. One, Copilot Studio's extensibility is bounded by what Microsoft exposes; the RAG pipeline gives us full control over retrieval, ranking, prompts, and observability. Two, the cost model is different — message-bundle pricing punishes us at scale; per-token pricing scales linearly.

**"How do we maintain this if you leave?"**
Reasonable concern given his transcript comments. The answer: this is ~600 lines of Python in a documented repo with eval coverage. Any mid-level Python developer can pick it up. Compare to Copilot Studio's GUI state which is captive in Microsoft. Hand them this document plus the repo and they're up in a week.

**"What's the security story?"**
KB never leaves your machine. Only the user's question and retrieved chunks go to Anthropic. Anthropic's API ToS plus enterprise tier covers data handling. No customer PII flows through the prototype because the demo uses mock tracking data. For production, the customer ID would flow only as a token to the tracking API; Claude would see the tracking *result*, not the PII.

**"What if Anthropic goes down?"**
The pipeline is provider-agnostic at the LLM layer. Swap Claude for OpenAI / Mistral / a self-hosted Llama with one config change in `llm.py`. Show this — it's a 5-line change.

**"Where's the eval methodology written down?"**
Section 12 of this bible. Hand him the doc.

---

# Part IV — Operations

## 17. Claude Code playbook

Claude Code is a CLI agent that can read your filesystem, run commands, edit files, and reason across the whole project. Use it as a senior pair-programmer who never gets tired. The trick is knowing when to use it and when not to.

### When to use Claude Code

**Project scaffolding (Phase 0).** Tell it the file structure from Section 6 and ask it to create the empty files and folders, populate `.gitignore`, write a starter `README.md`. This saves 30 minutes of typing.

Example prompt:
```
Create the project structure described in /path/to/PROJECT_BIBLE.md Section 6.
Create empty Python files with module docstrings only — no implementations.
Create .gitignore with the entries from Section 6's table.
Don't install anything yet.
```

**Boilerplate generation.** "Write the Pydantic model for an eval question matching the JSON schema in PROJECT_BIBLE.md Section 12." It produces clean code; you read it line by line and check it does what you want. Faster than typing, slower than copy-pasting from somewhere wrong.

**Test scaffolding.** "Write pytest tests for `loader.py` covering: empty file, file with one Item, file with five Items, file with no Items (should raise ValueError), file with malformed Item heading." It writes tests; you run them; you fix the loader if any fail. This is genuinely how senior engineers work.

**Eval runs.** "Run `python eval/run_eval.py` and show me the results CSV summary." It does it, you stay in the planning headspace.

**Debugging environment issues.** "I'm getting `ImportError: cannot import name 'X' from 'langchain'`. What's likely wrong?" It reads your requirements.txt, checks against the LangChain version, and tells you. Faster than Stack Overflow.

**Refactoring.** "Move all the prompt strings out of chain.py and into prompts.py, update imports." It does the boring work; you review the diff.

### When NOT to use Claude Code

**To write your retriever or chain end to end.** This is the work you need to do yourself. Reading working code you didn't write doesn't get you to fluency; only writing imperfect code, breaking it, and fixing it does.

**To generate the prompts.** System prompts encode product judgment ("APG tone", "no PayPal"). Write these yourself. Have Claude (in chat) review them, but don't ask Claude Code to author them.

**To make architecture decisions.** Decisions are this document's job. Claude Code's job is execution.

### Concrete commands you'll use

```bash
# Phase 0 setup
claude-code "Create the folder structure and empty files from PROJECT_BIBLE.md Section 6"

# Mid-project, after a session
claude-code "Run pytest and show me failures"
claude-code "Run python -m src.embedder and show me the output"

# Debugging
claude-code "Read logs/queries.jsonl and show me the 5 queries with longest latency"

# Eval
claude-code "Run python eval/run_eval.py with run_label='v2-hybrid'. Compare the results CSV against eval/results/v1-baseline.csv. Highlight questions where v2 changed score."
```

The commit messages it suggests are usually fine. Read them; tweak if needed.

---

## 18. Risks, kill criteria, and rollback

Risk register, ordered by impact × likelihood.

| # | Risk | Likelihood | Impact | Mitigation | Kill criterion |
|---|---|---|---|---|---|
| 1 | Eval target (27/33) not hit by Day 13 | Medium | High | Tune retriever weights, then prompts, then KB phrasing | If <22/33 on Day 13, demo as proof-of-concept rather than parity claim |
| 2 | Streamlit + streaming integration breaks late | Medium | Medium | Test streaming as soon as Session 10 starts; have non-streaming fallback ready | Keep `app_simple.py` (non-streaming) as a safety net |
| 3 | Anthropic API spend overrun | Low | Medium | Hard spend limit on Anthropic console set in Phase 0 | Cap is the cap. If hit, demo is delayed. |
| 4 | LangChain version pin breaks something Day 8+ | Medium | Medium | Pin lockfile after Phase 1 success; do not upgrade mid-project | Don't run `pip install -U` during the build window |
| 5 | KB has gaps (questions you can't answer regardless of bot quality) | High | Low | Categorise eval failures: separate "KB gap" from "bot bug" | Flag KB gaps to Ryan as separate finding, not a bot failure |
| 6 | Tool calling proves harder than expected | Medium | Medium | If Day 8 isn't done by end of Day 9, simplify to one tool, demo without escalation | Drop escalation tool if needed; mention as future work |
| 7 | Demo day blocked by laptop / network / Ryan's calendar | Medium | High | Recorded video backup from Session 14 | Send the video; reschedule live demo |
| 8 | APG security policy says no public GitHub | Low | Low | Repo is private; KB is gitignored | Move to GitLab / internal hosting if asked |

### Daily kill-criterion check

At the end of each day, take 10 minutes and ask: am I on track for the demo? If you're more than one day behind by Day 7, revise the scope. The order to drop features in: production-shape Docker (already excluded) → escalation tool → streaming UI (use non-streaming fallback) → reranking (use ensemble retriever alone). Do not drop hybrid search or memory — those are the "wow" features.

### Rollback plan

If at any point the build is in worse shape than the last working commit, hard-reset to that commit. Don't try to fix forward when the diff has grown. Git is your safety net specifically for this.

---

## 19. What this is NOT (be honest with Ryan and yourself)

It is not a production system. There is no auth, no rate limiting, no monitoring, no SLA, no horizontal scaling, no failover.

It is not integrated with APG's actual systems. The tracking tool is mocked. The escalation tool writes to a local JSON file, not the real ticketing system.

It does not handle voice, attachments, or images. Text-only, English-only.

It does not handle conversation lengths beyond a few turns gracefully. After ~10 turns, prompt cost grows and quality may degrade. A real deployment would summarise older history.

It has not had a security review. The Anthropic API ToS covers Anthropic's side; the rest is your laptop and your judgment.

It cannot be handed to a non-technical user as is. Streamlit on localhost is a developer demo, not a customer interface.

State these things in Section 4 of the demo. Better that Ryan hears them from you than asks for them.

---

## 20. Appendix — commands cheat sheet

### Project setup (Phase 0, one-time)

```bash
mkdir apg-rag-demo && cd apg-rag-demo
python -m venv venv
venv\Scripts\activate                              # Windows
# source venv/bin/activate                          # Mac/Linux
git init
# create .gitignore, .env, requirements.txt manually first
pip install -r requirements.txt
pip freeze > requirements.lock
git add .gitignore README.md requirements.txt requirements.lock
git commit -m "chore: project scaffold"
```

### Daily workflow

```bash
# Start of session
cd apg-rag-demo
venv\Scripts\activate
git checkout -b feat/sessionN-short-description    # one branch per session
# ... work ...
# End of session
pytest                                              # if applicable
git add -p                                          # interactively stage hunks
git commit -m "feat(retriever): add cross-encoder reranking"
git push origin feat/sessionN-short-description
# merge to main when session checkpoint passes
git checkout main && git merge feat/sessionN-short-description && git push
```

### Commit message conventions

`<type>(<scope>): <subject>`

Types: `feat` (new capability), `fix` (bug), `refactor` (no behaviour change), `docs`, `chore` (env, deps), `test`, `eval` (eval-related).

Examples:
- `feat(retriever): add cross-encoder reranking, +4 questions passing`
- `fix(loader): handle Items with subheadings without splitting`
- `eval(baseline): record v1 results, 24/33 passing`
- `docs(readme): add architecture diagram`

The "+N questions passing" annotation in commit subjects is gold for the demo — it shows your eval gating decisions over time.

### Useful one-liners during the build

```bash
# Tail logs in real time (separate terminal)
tail -f logs/queries.jsonl | jq .

# Count vectors in an index
python -c "from langchain_community.vectorstores import FAISS; from langchain_huggingface import HuggingFaceEmbeddings; idx = FAISS.load_local('indexes/thresholds', HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5'), allow_dangerous_deserialization=True); print(idx.index.ntotal)"

# Wipe and rebuild all indexes
rm -rf indexes/ && python -m src.embedder

# Quick eval run
python eval/run_eval.py --run_label="$(date +%Y%m%d-%H%M)"
```

### When something breaks

The diagnostic order, in time it takes:
1. Read the error message slowly. Twice.
2. Check the log file for the most recent query — what was retrieved? what was the prompt?
3. Read the stack trace from the bottom up — find the first line that's *your* code.
4. Re-run with one input you've debugged before, to isolate whether the input changed or the code changed.
5. If still stuck, paste the error + the relevant code into Claude (chat) and ask "what's most likely wrong here". Don't ask Claude to fix it; ask for the *most likely cause* and fix it yourself.

The discipline of step 5 is what makes the difference between using AI to learn and using AI to avoid learning.

---

## Closing

Read this document end to end before you do anything. Then come back and we start Session 1.

When something here is wrong or unclear, tell me. The bible is a living document until Day 14, then it's archived.

Make Ryan's life easier than Microsoft's been making it.
