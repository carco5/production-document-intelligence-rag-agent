from app.core.settings import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_markdown_documents
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.schemas import RetrievedChunk
from app.retrieval.vector_store import InMemoryVectorStore


class LocalRetriever:
    def __init__(self) -> None:
        self.embedder = LocalEmbedder()
        self.store = InMemoryVectorStore()
        self._is_indexed = False

    def index_documents(self) -> None:
        documents = load_markdown_documents(settings.data_path)
        chunks = chunk_documents(
            documents=documents,
            chunk_size=220,
            chunk_overlap=40,
        )
        embeddings = self.embedder.embed_texts([chunk.content for chunk in chunks])
        self.store.add(chunks=chunks, embeddings=embeddings)
        self._is_indexed = True

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not self._is_indexed:
            self.index_documents()

        query_embedding = self.embedder.embed_query(query)
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k or settings.top_k,
        )