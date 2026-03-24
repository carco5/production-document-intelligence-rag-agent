from app.ingestion.schemas import DocumentChunk, LoadedDocument


def _find_split_end(
    text: str,
    start: int,
    max_end: int,
    min_chunk_size: int,
) -> int:
    if max_end >= len(text):
        return len(text)

    window = text[start:max_end]
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " "]

    for separator in separators:
        idx = window.rfind(separator)
        if idx >= min_chunk_size:
            return start + idx + len(separator)

    return max_end


def _adjust_start_forward(text: str, start: int) -> int:
    if start <= 0:
        return 0

    if start >= len(text):
        return len(text)

    if text[start].isspace():
        while start < len(text) and text[start].isspace():
            start += 1
        return start

    while start < len(text) and not text[start].isspace():
        start += 1

    while start < len(text) and text[start].isspace():
        start += 1

    return start


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 220,
    chunk_overlap: int = 40,
    min_chunk_size: int = 80,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be greater than 0")

    chunks: list[DocumentChunk] = []

    for doc in documents:
        text = doc.content.strip()
        start = 0
        chunk_index = 0

        while start < len(text):
            max_end = min(start + chunk_size, len(text))
            end = _find_split_end(
                text=text,
                start=start,
                max_end=max_end,
                min_chunk_size=min_chunk_size,
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.path.stem}_chunk_{chunk_index}",
                        source=doc.source,
                        title=doc.title,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                    )
                )

            if end >= len(text):
                break

            raw_next_start = max(end - chunk_overlap, 0)
            next_start = _adjust_start_forward(text, raw_next_start)

            if next_start <= start or next_start >= len(text):
                next_start = end

            start = next_start
            chunk_index += 1

    return chunks