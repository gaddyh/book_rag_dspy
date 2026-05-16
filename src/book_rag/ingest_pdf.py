import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

# Add src to Python path if not already there
src_path = Path("src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from book_rag.chunker import TextChunk, chunk_text


# Get project root (script is in src/book_rag/, so go up 2 levels)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_PDF_PATH = PROJECT_ROOT / "data" / "raw" / "30_agents.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

def extract_chunks_from_pdf(pdf_path: Path) -> list[TextChunk]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_chunks: list[TextChunk] = []

    source_name = pdf_path.name

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        text = page.get_text()

        page_chunks = chunk_text(
            text=text,
            source=source_name,
            page_number=page_number,
        )

        all_chunks.extend(page_chunks)

    return all_chunks


def save_chunks(chunks: list[TextChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def main() -> None:
    chunks = extract_chunks_from_pdf(RAW_PDF_PATH)
    save_chunks(chunks, OUTPUT_PATH)

    print(f"PDF: {RAW_PDF_PATH}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    if chunks:
        print("\nFirst chunk preview:")
        print("-" * 80)
        print(chunks[0].text[:700])
        print("-" * 80)
        print(chunks[0])


if __name__ == "__main__":
    main()
