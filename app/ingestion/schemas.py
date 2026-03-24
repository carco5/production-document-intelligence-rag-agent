from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    source: str
    title: str
    content: str
    path: Path


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int