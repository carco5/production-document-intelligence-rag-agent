from app.agent.graph import build_rag_graph
from app.rag.schemas import RAGResponse, RetrievedChunkResult


class LangGraphRAGService:
    def __init__(self) -> None:
        self.graph = build_rag_graph()

    def query(self, user_query: str) -> RAGResponse:
        result = self.graph.invoke({"query": user_query})

        retrieved_chunks = result.get("retrieved_chunks", [])
        sources = result.get("sources", [])
        answer = result.get("answer", "")

        chunk_results = [
            RetrievedChunkResult(
                chunk_id=item.chunk.chunk_id,
                source=item.chunk.source,
                title=item.chunk.title,
                content=item.chunk.content,
                score=item.score,
            )
            for item in retrieved_chunks
        ]

        return RAGResponse(
            query=user_query,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunk_results,
        )