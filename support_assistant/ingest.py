from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append({
            "id": path.stem,
            "document": text,
            "metadata": {
                "doc_id": path.stem,
                "source": path.name,
            },
        })

    if len(documents) != 8:
        raise ValueError(f"Expected 8 policy documents, found {len(documents)}")

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
        metadata={"description": "Zepto policy documents"},
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

    print(f"Indexed {collection.count()} documents.")
    print(f"Chroma database: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
