from app.core.settings import settings
from app.ingestion.loader import load_markdown_documents


def main() -> None:
    documents = load_markdown_documents(settings.data_path)

    print(f"Loaded documents: {len(documents)}")
    print("-" * 60)

    for idx, doc in enumerate(documents, start=1):
        preview = doc.content[:200].replace("\n", " ")
        print(f"[{idx}] {doc.title}")
        print(f"    source: {doc.source}")
        print(f"    chars: {len(doc.content)}")
        print(f"    preview: {preview}")
        print("-" * 60)


if __name__ == "__main__":
    main()