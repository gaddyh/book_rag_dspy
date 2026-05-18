import json
from pathlib import Path

import fitz  # PyMuPDF

from book_rag.chunker import chunk_text
from book_rag.core.models import BookChunk


RAW_PDF_PATH = Path("data/raw/30_agents.pdf")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")


def count_visual_image_blocks(page: fitz.Page) -> int:
    """
    Count visual image blocks on a page.

    This is better than page.get_images(full=True), because get_images()
    can count reused/internal PDF image resources and produce misleading
    numbers like 134 images on many pages.
    """
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])

    return sum(1 for block in blocks if block.get("type") == 1)


def extract_chunks_from_pdf(pdf_path: Path) -> list[BookChunk]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    all_chunks: list[BookChunk] = []

    source_name = pdf_path.name

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        text = page.get_text()

        image_count = count_visual_image_blocks(page)
        has_images = image_count > 0

        page_chunks = chunk_text(
            text=text,
            source=source_name,
            page_number=page_number,
            has_images=has_images,
            image_count=image_count,
        )

        all_chunks.extend(page_chunks)

    return all_chunks


def save_chunks(chunks: list[BookChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def main() -> None:
    chunks = extract_chunks_from_pdf(RAW_PDF_PATH)
    save_chunks(chunks, OUTPUT_PATH)

    print(f"PDF: {RAW_PDF_PATH.resolve()}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")

    if chunks:
        print("\nFirst chunk preview:")
        print("-" * 80)
        print(chunks[0].text[:700])
        print("-" * 80)
        print(chunks[0])


if __name__ == "__main__":
    main()