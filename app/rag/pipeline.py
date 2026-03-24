from app.generation.ollama_generator import OllamaRAGGenerator
from app.rag.schemas import RAGResponse, RetrievedChunkResult
from app.retrieval.retriever import LocalRetriever


class LocalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = LocalRetriever()
        self.generator = OllamaRAGGenerator()

    def run(self, query: str) -> RAGResponse:
        retrieved_chunks = self.retriever.search(query=query)

        answer = self.generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        sources = sorted({item.chunk.source for item in retrieved_chunks})

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
            query=query,
            answer=answer,
            sources=sources,
            retrieved_chunks=chunk_results,
        )