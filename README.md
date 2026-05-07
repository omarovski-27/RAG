# RAG

Local retrieval-augmented generation prototype for an APG-style customer support chatbot.

This repository currently contains the project specification, dependency baseline, and architecture diagrams for a local RAG build that combines hybrid retrieval, reranking, tool use, and a lightweight chat UI.

## Current scope

- `PROJECT_BIBLE.md` is the working specification for architecture, build phases, evaluation, and demo flow.
- `requirements.txt` captures the initial Python dependency set for the prototype.
- `diagrams/` contains the draw.io source files and supporting diagram source notes.

## Planned architecture

- Local document loading and indexing
- Hybrid retrieval with BM25 and FAISS
- Cross-encoder reranking
- Anthropic-powered answer generation with tool calling
- Streamlit chat interface
- Structured JSON logging and evaluation harness

## Repository layout

```text
.
├── PROJECT_BIBLE.md
├── README.md
├── requirements.txt
└── diagrams/
```

## Notes

- Knowledge-base content, generated indexes, logs, environment files, and local virtual environments are intentionally excluded from version control.
- This repo is currently a scaffold and design baseline; the implementation modules described in the project bible are planned next.

## Getting started

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Review `PROJECT_BIBLE.md` before building the implementation modules.

## Diagrams

The architecture and flow diagrams live under `diagrams/` as `.drawio` files so they can be edited in draw.io Desktop and versioned as source.