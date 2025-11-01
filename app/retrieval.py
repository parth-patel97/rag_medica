# app/retrieval.py
import os
from typing import List
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from app.logging import get_logger

logger = get_logger()

_CHROMA_PERSIST_DIR = os.path.abspath(os.environ.get("CHROMA_PERSIST_DIR", "models/chroma"))
_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "collection_sample")
_EMBED_MODEL = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

_embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)

def _load_vectorstore(persist_directory: str = _CHROMA_PERSIST_DIR, collection_name: str = _COLLECTION_NAME):
    try:
        vect = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=_embeddings)
        logger.info("✅ Chroma vectorstore loaded via LangChain.")
        return vect
    except Exception:
        logger.exception("❌ Failed to load Chroma vectorstore")
        raise

def _format_hit(doc, score, idx):
    meta = doc.metadata or {}
    return {
        "id": meta.get("id", idx),
        "score": float(score),
        "text": doc.page_content,
        "page": meta.get("page"),
        "size": meta.get("size"),
    }

def retrieve(query: str, top_k: int = 5) -> List[dict]:
    try:
        vect = _load_vectorstore()
        docs_and_scores = vect.similarity_search_with_score(query, k=top_k)
        hits = [_format_hit(doc, score, i) for i, (doc, score) in enumerate(docs_and_scores)]
        logger.info(f"✅ Retrieved top-{top_k} chunks for query: {query}")
        return hits
    except Exception:
        logger.exception(f"❌ Retrieval failed for query: {query}")
        return []
