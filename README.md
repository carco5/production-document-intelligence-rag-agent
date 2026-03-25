# Production Document Intelligence RAG Agent

A modular, production-oriented Retrieval-Augmented Generation (RAG) system built with FastAPI, LangGraph, and local LLM inference (Ollama).

---

## Overview

This project implements an end-to-end RAG pipeline designed with production best practices:

* Document ingestion and chunking
* Semantic retrieval using embeddings
* Context-aware answer generation
* Agent-based orchestration with LangGraph
* Local LLM integration via Ollama
* API exposure with FastAPI
* Containerization with Docker
* Observability through structured logging

The system is designed to be scalable, explainable, and easily extensible.

---

## Architecture

The pipeline follows a modular architecture:

```text
User Query
    ↓
Retriever (Embeddings + Vector Search)
    ↓
Relevance Filtering (Threshold-based routing)
    ↓
LangGraph Agent
    ├── Generate Answer (LLM)
    └── No Context (Fallback)
    ↓
API Response (FastAPI)
```

---

## Key Features

### 1. Semantic Retrieval

* Uses `sentence-transformers` (MiniLM)
* Cosine similarity via normalized embeddings
* In-memory vector store

### 2. Intelligent Chunking

* Context-aware splitting using sentence boundaries
* Overlap strategy to preserve semantic continuity

### 3. Relevance-Aware Routing

* Threshold-based decision (`top_score`)
* Prevents hallucinations when context is weak

### 4. Local LLM Integration

* Powered by Ollama (`llama3.2:3b`)
* Fully local inference (no external API dependency)

### 5. Agent-Based Orchestration

* Implemented with LangGraph
* Conditional execution:

  * `generate` if enough context
  * `no_context` otherwise

### 6. Observability (Logging)

* Tracks:

  * query
  * number of retrieved chunks
  * top similarity score
  * routing decision

Example:

```text
[INFO] [RETRIEVE] query='...' | chunks=3 | top_score=0.6252 | status=enough_context
[WARNING] [NO_CONTEXT] query='...' | top_score=0.2388
```

### 7. API Layer

* Built with FastAPI
* Endpoints:

  * `GET /health`
  * `POST /query`

### 8. Containerization

* Dockerized application
* Ready for deployment or orchestration (e.g., Kubernetes)

---

## Project Structure

```text
app/
├── agent/          # LangGraph agent logic
├── api/            # FastAPI endpoints and schemas
├── core/           # Config and logging
├── generation/     # LLM generators (baseline + Ollama)
├── ingestion/      # Document loading and chunking
├── rag/            # Pipeline and response schemas
├── retrieval/      # Embeddings and vector store
├── services/       # Service layer abstraction

data/
└── knowledge_base/ # Markdown documents

scripts/
└── test_*          # Testing utilities
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run locally

```bash
uvicorn app.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

### 3. Run with Docker

```bash
docker build -t rag-agent .
docker run --rm -p 8000:8000 rag-agent
```

---

### 4. Run with Docker Compose

```bash
docker compose up --build
```

---

## Local LLM Setup (Ollama)

Install Ollama and pull the model:

```bash
ollama pull llama3.2:3b
```

The system connects to:

```text
http://127.0.0.1:11434
```

---

## Example Query

```json
POST /query

{
  "query": "What are embeddings used for in RAG systems?"
}
```

---

## Example Behavior

### High relevance

* Retrieves relevant chunks
* Generates grounded answer

### Low relevance

* Skips generation
* Returns fallback with explanation

---

## Design Principles

* Modularity (clear separation of concerns)
* Observability (logging and traceability)
* Reliability (avoid hallucinations)
* Extensibility (easy to swap components)
* Production-readiness (Docker, API, agent orchestration)

---

## Future Improvements

* Persistent vector database (FAISS / Qdrant)
* Streaming responses
* Multi-step reasoning agents
* Query rewriting
* Hybrid search (BM25 + embeddings)
* Kubernetes deployment

---

## Tech Stack

* Python
* FastAPI
* LangGraph
* Sentence Transformers
* Ollama
* NumPy
* Docker

---

## Author

Josep
Data & AI Engineering

---
