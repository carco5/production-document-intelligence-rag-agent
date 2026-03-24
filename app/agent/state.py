from typing import TypedDict

from app.retrieval.schemas import RetrievedChunk


class RAGAgentState(TypedDict, total=False):
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    sources: list[str]