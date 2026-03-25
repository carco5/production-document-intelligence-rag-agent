from typing import Literal, TypedDict

from app.retrieval.schemas import RetrievedChunk


class RAGAgentState(TypedDict, total=False):
    query: str
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    sources: list[str]
    retrieval_status: Literal["enough_context", "insufficient_context"]
    top_score: float