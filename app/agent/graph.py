from langgraph.graph import END, StateGraph

from app.agent.state import RAGAgentState
from app.generation.ollama_generator import OllamaRAGGenerator
from app.retrieval.retriever import LocalRetriever


retriever = LocalRetriever()
generator = OllamaRAGGenerator()


def retrieve_node(state: RAGAgentState) -> RAGAgentState:
    query = state["query"]
    retrieved_chunks = retriever.search(query=query)
    sources = sorted({item.chunk.source for item in retrieved_chunks})

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
    }


def generate_node(state: RAGAgentState) -> RAGAgentState:
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])

    answer = generator.generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        **state,
        "answer": answer,
    }


def build_rag_graph():
    graph = StateGraph(RAGAgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()