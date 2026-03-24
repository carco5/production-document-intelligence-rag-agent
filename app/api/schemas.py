from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    source: str
    title: str
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunkResponse]