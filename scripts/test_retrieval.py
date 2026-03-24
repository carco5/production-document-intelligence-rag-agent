from app.core.settings import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_markdown_documents
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import InMemoryVectorStore


def main() -> None:
    query = "What are embeddings used for in RAG systems?"

    documents = load_markdown_documents(settings.data_path)
    chunks = chunk_documents(documents=documents, chunk_size=220, chunk_overlap=40)

    embedder = LocalEmbedder()
    chunk_embeddings = embedder.embed_texts([chunk.content for chunk in chunks])

    store = InMemoryVectorStore()
    store.add(chunks=chunks, embeddings=chunk_embeddings)

    query_embedding = embedder.embed_query(query)
    results = store.search(query_embedding=query_embedding, top_k=settings.top_k)

    print(f"Query: {query}")
    print(f"Indexed chunks: {len(chunks)}")
    print("-" * 60)

    for idx, result in enumerate(results, start=1):
        preview = result.chunk.content[:180].replace("\n", " ")
        print(f"[{idx}] score={result.score:.4f}")
        print(f"    chunk_id: {result.chunk.chunk_id}")
        print(f"    source: {result.chunk.source}")
        print(f"    title: {result.chunk.title}")
        print(f"    preview: {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()