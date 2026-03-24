from app.core.settings import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_markdown_documents


def main() -> None:
    documents = load_markdown_documents(settings.data_path)
    chunks = chunk_documents(documents=documents, chunk_size=220, chunk_overlap=40)

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print("-" * 60)

    for idx, chunk in enumerate(chunks, start=1):
        preview = chunk.content[:160].replace("\n", " ")
        print(f"[{idx}] {chunk.chunk_id}")
        print(f"    source: {chunk.source}")
        print(f"    title: {chunk.title}")
        print(f"    chunk_index: {chunk.chunk_index}")
        print(f"    span: {chunk.start_char}:{chunk.end_char}")
        print(f"    chars: {len(chunk.content)}")
        print(f"    preview: {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()