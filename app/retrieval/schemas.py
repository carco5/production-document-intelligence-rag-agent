from dataclasses import dataclass

from app.ingestion.schemas import DocumentChunk


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float