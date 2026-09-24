from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a policy document into overlapping text chunks."""
    text = " ".join(text.split())

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        for index, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{path.stem}_chunk_{index}",
                    "document": chunk,
                    "metadata": {
                        "doc_id": path.stem,
                        "chunk_id": f"{path.stem}_chunk_{index}",
                        "source": path.name,
                    },
                }
            )

    if len(list(DOCS_DIR.glob("doc_*.txt"))) != 8:
        raise ValueError("Expected exactly 8 policy documents.")

    return documents


def main():
    documents = load_documents()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Zepto policy document chunks"},
    )

    texts = [item["document"] for item in documents]
    ids = [item["id"] for item in documents]
    metadatas = [item["metadata"] for item in documents]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {collection.count()} document chunks.")
    print(f"Chroma database: {CHROMA_DIR}")


if __name__ == "__main__":
    main()