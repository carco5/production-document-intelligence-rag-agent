from app.generation.generator import BaselineRAGGenerator
from app.retrieval.retriever import LocalRetriever


class LocalRAGPipeline:
    def __init__(self) -> None:
        self.retriever = LocalRetriever()
        self.generator = BaselineRAGGenerator()

    def run(self, query: str) -> dict:
        retrieved_chunks = self.retriever.search(query=query)

        answer = self.generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
        }