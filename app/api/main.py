from fastapi import FastAPI

from app.api.schemas import QueryRequest, QueryResponse, RetrievedChunkResponse
from app.services.rag_service import RAGService

app = FastAPI(
    title="Production Document Intelligence RAG Agent",
    version="0.1.0",
)

rag_service = RAGService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    result = rag_service.query(user_query=request.query)

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=item.chunk_id,
                source=item.source,
                title=item.title,
                content=item.content,
                score=item.score,
            )
            for item in result.retrieved_chunks
        ],
    )