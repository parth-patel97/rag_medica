from app.ingestion import dynamic_chunk_text_with_variance
from app.embeddings import build_index
from app.logging import get_logger

logger = get_logger()

if __name__ == "__main__":
    try:
        chunks = dynamic_chunk_text_with_variance("10050-medicare-and-you.pdf")
        build_index(chunks)
    except Exception as e:
        logger.exception("❌ Index building failed.")
