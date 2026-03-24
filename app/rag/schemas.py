from dataclasses import dataclass


@dataclass
class RetrievedChunkResult:
    chunk_id: str
    source: str
    title: str
    content: str
    score: float


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunkResult]