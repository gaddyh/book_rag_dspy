from book_rag.core.models import BookChunk, RetrievedContext


class ChunkOnlyContextBuilder:
    strategy = "chunk_only"

    def build(self, chunks: list[BookChunk]) -> RetrievedContext:
        context_parts = []

        for chunk in chunks:
            if chunk.page_start is not None and chunk.page_end is not None and chunk.page_end != chunk.page_start:
                page_label = f"pages {chunk.page_start}-{chunk.page_end}"
            elif chunk.page_start is not None:
                page_label = f"page {chunk.page_start}"
            else:
                page_label = "page unknown"

            context_parts.append(
                f"[{chunk.chunk_id} | {page_label} | images: {chunk.has_images} ({chunk.image_count})]\n"
                f"{chunk.text}"
            )

        text = "\n\n---\n\n".join(context_parts)

        return RetrievedContext(
            text=text,
            chunks=chunks,
            strategy=self.strategy,
        )
