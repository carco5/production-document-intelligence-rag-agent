from langgraph.graph import END, StateGraph

from app.agent.state import RAGAgentState
from app.generation.ollama_generator import OllamaRAGGenerator
from app.retrieval.retriever import LocalRetriever
from app.core.logger import setup_logger

logger = setup_logger()


retriever = LocalRetriever()
generator = OllamaRAGGenerator()

MIN_RELEVANCE_SCORE = 0.55


def retrieve_node(state: RAGAgentState) -> RAGAgentState:
    query = state["query"]
    retrieved_chunks = retriever.search(query=query)
    sources = sorted({item.chunk.source for item in retrieved_chunks})

    top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0

    retrieval_status = (
        "enough_context"
        if retrieved_chunks and top_score >= MIN_RELEVANCE_SCORE
        else "insufficient_context"
    )

    logger.info(
        f"[RETRIEVE] query='{query}' | chunks={len(retrieved_chunks)} | top_score={top_score:.4f} | status={retrieval_status}"
    )

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "retrieval_status": retrieval_status,
        "top_score": top_score,
    }


def route_after_retrieval(state: RAGAgentState) -> str:
    return state.get("retrieval_status", "insufficient_context")


def generate_node(state: RAGAgentState) -> RAGAgentState:
    query = state["query"]
    retrieved_chunks = state.get("retrieved_chunks", [])

    logger.info(f"[GENERATE] query='{query}' | using {len(retrieved_chunks)} chunks")

    answer = generator.generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        **state,
        "answer": answer,
    }


def no_context_node(state: RAGAgentState) -> RAGAgentState:
    top_score = state.get("top_score", 0.0)

    logger.warning(
        f"[NO_CONTEXT] query='{state.get('query')}' | top_score={top_score:.4f}"
    )

    return {
        **state,
        "answer": (
            "I could not find enough relevant context in the knowledge base to answer "
            f"this question reliably. Top retrieval score: {top_score:.4f}"
        ),
    }


def build_rag_graph():
    graph = StateGraph(RAGAgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_context", no_context_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "enough_context": "generate",
            "insufficient_context": "no_context",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph.compile()