from app.retrieval.schemas import RetrievedChunk


class BaselineRAGGenerator:
    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return (
                "I could not find relevant context in the knowledge base to answer "
                "this question."
            )

        bullet_points: list[str] = []

        for item in retrieved_chunks:
            text = " ".join(item.chunk.content.split())
            bullet_points.append(f"- {text}")

        response = [
            f"Question: {query}",
            "",
            "Answer based on retrieved context:",
            *bullet_points,
        ]

        return "\n".join(response)