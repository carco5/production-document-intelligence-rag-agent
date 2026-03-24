from app.agent.graph import build_rag_graph


def main() -> None:
    graph = build_rag_graph()

    query = "What are embeddings used for in RAG systems?"

    result = graph.invoke({"query": query})

    print("=" * 80)
    print(f"Query: {result['query']}")
    print("-" * 80)
    print(result["answer"])
    print("=" * 80)
    print(f"Sources: {', '.join(result.get('sources', []))}")
    print("=" * 80)
    print("Retrieved chunks:")
    print("-" * 80)

    for idx, item in enumerate(result.get("retrieved_chunks", []), start=1):
        preview = item.chunk.content[:180].replace("\n", " ")
        print(f"[{idx}] score={item.score:.4f}")
        print(f"    chunk_id: {item.chunk.chunk_id}")
        print(f"    source: {item.chunk.source}")
        print(f"    title: {item.chunk.title}")
        print(f"    preview: {preview}")
        print("-" * 80)


if __name__ == "__main__":
    main()