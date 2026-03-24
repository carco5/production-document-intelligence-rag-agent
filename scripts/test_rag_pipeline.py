from app.rag.pipeline import LocalRAGPipeline


def main() -> None:
    query = "What are embeddings used for in RAG systems?"

    pipeline = LocalRAGPipeline()
    result = pipeline.run(query=query)

    print("=" * 80)
    print(f"Query: {result.query}")
    print("-" * 80)
    print(result.answer)
    print("=" * 80)
    print(f"Sources: {', '.join(result.sources)}")
    print("=" * 80)
    print("Retrieved chunks:")
    print("-" * 80)

    for idx, item in enumerate(result.retrieved_chunks, start=1):
        preview = item.content[:180].replace("\n", " ")
        print(f"[{idx}] score={item.score:.4f}")
        print(f"    chunk_id: {item.chunk_id}")
        print(f"    source: {item.source}")
        print(f"    title: {item.title}")
        print(f"    preview: {preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()