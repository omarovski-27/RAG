"""
loader.py — Load APG KB markdown files into LangChain Documents.

Splits at ## Item_NNN boundaries only (not any ## heading).
Each Document gets metadata: source_file, topic, item_id, item_title.
"""
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Matches exactly '## Item_' at the start of a line, captures id + title
_ITEM_PATTERN = re.compile(r"^## (Item_\d+):\s*(.+)$", re.MULTILINE)


def _derive_topic(file_path: Path) -> str:
    """
    Turn a KB filename into a short topic slug.
    'duties_thresholds_by_country.md' → 'thresholds'
    'wismo.md'                        → 'wismo'
    'damaged_goods.md'                → 'damaged_goods'
    """
    stem = file_path.stem  # filename without extension
    if stem.startswith("duties_"):
        return stem[len("duties_"):]
    return stem


def load_kb_file(file_path: Path) -> List[Document]:
    """
    Read a single KB markdown file and split into Documents at Item boundaries.

    Each Document corresponds to one ## Item_NNN block. Metadata:
      - source_file : filename string
      - topic       : derived from filename
      - item_id     : e.g. 'Item_004'
      - item_title  : heading text after the colon

    Raises ValueError if no Items are found (likely wrong file).
    """
    text = file_path.read_text(encoding="utf-8")
    matches = list(_ITEM_PATTERN.finditer(text))

    if not matches:
        raise ValueError(f"No ## Item_NNN headings found in {file_path.name}")

    topic = _derive_topic(file_path)
    docs: List[Document] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()

        docs.append(Document(
            page_content=chunk,
            metadata={
                "source_file": file_path.name,
                "topic":       topic,
                "item_id":     match.group(1),
                "item_title":  match.group(2).strip(),
            },
        ))

    return docs


def load_all_kb_files(kb_dir: Path) -> dict[str, List[Document]]:
    """
    Load every .md file in kb_dir.
    Returns a dict mapping topic slug → list of Documents.
    """
    result: dict[str, List[Document]] = {}
    for md_file in sorted(kb_dir.glob("*.md")):
        docs = load_kb_file(md_file)
        topic = _derive_topic(md_file)
        result[topic] = docs
    return result


if __name__ == "__main__":
    # Quick smoke-test: python -m src.loader
    from src.config import KB_DIR

    all_docs = load_all_kb_files(KB_DIR)
    for topic, docs in all_docs.items():
        print(f"\n── {topic} ({len(docs)} items) ──")
        for doc in docs:
            print(f"  [{doc.metadata['item_id']}] {doc.metadata['item_title']}")
