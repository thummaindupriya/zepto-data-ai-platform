from pathlib import Path
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from support_assistant.prompts import PROMPT_TEMPLATE


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0, le=1)


app = FastAPI(title="Zepto Support Assistant")


_embedding_model = None
_collection = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def get_collection():
    global _collection

    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)

    return _collection


def classify_intent(state: GraphState):
    query = state["query"].lower()

    if any(keyword in query for keyword in KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: GraphState):
    query = state["query"]

    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return {
            "answer": "I could not find relevant information in the Zepto policy documents.",
            "sources": [],
            "confidence": 0.0,
        }

    context_parts = []

    for metadata, document in zip(metadatas, documents):
        context_parts.append(
            f"{metadata['doc_id']}: {document}"
        )

    context = "\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=context,
    )

    # MOCK_LLM baseline:
    # deterministic answer from the highest-ranked retrieved chunk.
    del prompt

    top_chunk = documents[0]
    top_source = metadatas[0]["doc_id"]

    snippet = top_chunk[:200]

    answer = f"Based on the retrieved context: {snippet}"

    return {
        "answer": answer,
        "sources": [metadata["doc_id"] for metadata in metadatas],
        "confidence": 1.0,
    }


def direct_answer(state: GraphState):
    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0,
    }


def route_intent(state: GraphState):
    return state["intent"]


workflow = StateGraph(GraphState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})

    return AskResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
    )
