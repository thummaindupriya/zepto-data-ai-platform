# Zepto Support Assistant

A local Zepto policy question-answering assistant built with ChromaDB, Sentence Transformers, LangGraph, Pydantic, and FastAPI.

## Architecture

The system follows this flow:

```text
8 Zepto Policy Documents
        |
        v
    ingest.py
        |
        +--> Load documents
        |
        +--> Chunk text
        |    chunk_size=500
        |    overlap=50
        |
        v
SentenceTransformer
all-MiniLM-L6-v2
        |
        v
     ChromaDB
collection: zepto_policies
        |
        v
    FastAPI /ask
        |
        v
LangGraph StateGraph
        |
        +--> classify_intent
        |        |
        |        +--> policy_question
        |        |        |
        |        |        v
        |        |  retrieve_context
        |        |        |
        |        |        v
        |        |  build_prompt
        |        |        |
        |        |        v
        |        |  retrieve_and_answer
        |        |
        |        +--> general_question
        |                 |
        |                 v
        |           direct_answer
        |
        v
Pydantic validated response