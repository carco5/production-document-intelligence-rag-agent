from app.rag.pipeline import LocalRAGPipeline
from app.rag.schemas import RAGResponse


class RAGService:
    def __init__(self) -> None:
        self.pipeline = LocalRAGPipeline()

    def query(self, user_query: str) -> RAGResponse:
        return self.pipeline.run(query=user_query)