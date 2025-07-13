import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.logging import get_logger

logger = get_logger()
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks, index_path="models/faiss.idx", meta_path="models/meta.pkl"):
    """
    Builds FAISS index and saves metadata.

    Args:
        chunks (List[dict]): List of chunks with text, page, and size.
        index_path (str): Filepath to save FAISS index.
        meta_path (str): Filepath to save chunk metadata.
    """
    try:
        embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, index_path)

        with open(meta_path, "wb") as f:
            pickle.dump(chunks, f)

        logger.info(f"✅ FAISS index and metadata saved. Total chunks: {len(chunks)}")

    except Exception as e:
        logger.exception(f"❌ Failed to build FAISS index: {e}")
