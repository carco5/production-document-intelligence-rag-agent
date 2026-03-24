import numpy as np

from app.ingestion.schemas import DocumentChunk
from app.retrieval.schemas import RetrievedChunk


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        if not chunks:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[RetrievedChunk]:
        if self.embeddings is None or not self.chunks:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)

        scores = self.embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(idx)],
                    score=float(scores[int(idx)]),
                )
            )

        return results