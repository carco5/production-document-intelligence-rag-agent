from ollama import Client

from app.retrieval.schemas import RetrievedChunk


class OllamaRAGGenerator:
    def __init__(self, model_name: str = "llama3.2:3b") -> None:
        self.model_name = model_name
        self.client = Client(host="http://127.0.0.1:11434")

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I could not find relevant context in the knowledge base to answer this question."

        context = "\n\n".join(
            f"[Source: {item.chunk.source}]\n{item.chunk.content}"
            for item in retrieved_chunks
        )

        prompt = f"""You are a helpful AI assistant.

Answer the user's question using only the retrieved context below.
If the answer is not supported by the context, say so clearly.
Be concise, accurate, and grounded in the provided information.

Retrieved context:
{context}

User question:
{query}

Answer:
"""

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return response["message"]["content"].strip()