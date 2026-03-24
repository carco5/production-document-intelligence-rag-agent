from pathlib import Path

from app.ingestion.schemas import LoadedDocument


def load_markdown_documents(data_dir: Path) -> list[LoadedDocument]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[LoadedDocument] = []

    for path in sorted(data_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            LoadedDocument(
                source=path.name,
                title=path.stem.replace("_", " ").title(),
                content=content,
                path=path,
            )
        )

    return documents