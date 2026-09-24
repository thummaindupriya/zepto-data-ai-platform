# Zepto Support Assistant

A local Zepto policy question-answering assistant built with ChromaDB, Sentence Transformers, LangGraph, Pydantic, and FastAPI.

## Architecture

```text
8 Zepto Policy Documents
        |
        v
    ingest.py
        |
        v
SentenceTransformer
all-MiniLM-L6-v2
        |
        v
     ChromaDB
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
        |        |  retrieve_and_answer
        |        |
        |        +--> general_question
        |                 |
        |                 v
        |           direct_answer
        |
        v
Pydantic validated response
