# app/embeddings.py
import os
from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from app.logging import get_logger

logger = get_logger()
_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

_embeddings = HuggingFaceEmbeddings(model_name=_MODEL)

def get_embeddings(texts: List[str]):
    """Return a list of embeddings for the provided texts."""
    try:
        return _embeddings.embed_documents(texts)
    except Exception:
        logger.exception("Failed to compute embeddings")
        raise

def get_embedding(text: str):
    try:
        return _embeddings.embed_query(text)
    except Exception:
        logger.exception("Failed to compute embedding")
        raise
