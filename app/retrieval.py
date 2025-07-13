import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.logging import get_logger

logger = get_logger()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    index = faiss.read_index("models/faiss.idx")
    with open("models/meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    logger.info("✅ FAISS index and metadata loaded.")
except Exception as e:
    logger.exception("❌ Error loading FAISS index or metadata.")

def retrieve(query, top_k=5):
    """
    Retrieve top-k relevant chunks for a given query.

    Args:
        query (str): User query.
        top_k (int): Number of chunks to return.

    Returns:
        List[dict]: Top-k matched chunks with metadata and similarity scores.
    """
    try:
        q_emb = embed_model.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k * 3)

        hits = [{"id": i, "score": float(D[0][j]), **metadata[i]} for j, i in enumerate(I[0])]
        top_hits = hits[:top_k]

        logger.info(f"✅ Retrieved top-{top_k} chunks for query: {query}")
        return top_hits

    except Exception as e:
        logger.exception(f"❌ Retrieval failed for query: {query}")
        return []
