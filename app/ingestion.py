# app/ingestion.py
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from app.logging import get_logger

logger = get_logger()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def dynamic_chunk_text_with_variance(pdf_path, target_tokens=512, overlap_pct=0.2):
    """
    Chunk a PDF document into dynamically sized chunks based on local embedding variance.

    Args:
        pdf_path (str): Path to the PDF file.
        target_tokens (int): Average target chunk size in tokens.
        overlap_pct (float): Overlap percentage between consecutive chunks.

    Returns:
        List[dict]: A list of chunk dictionaries with page, text, and token size.
    """
    chunks = []

    try:
        doc = fitz.open(pdf_path)
        tokenizer = embed_model.tokenizer

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue

            sentences = text.split(". ")
            if len(sentences) < 2:
                logger.warning(f"Skipping page {page_num}: too few sentences.")
                continue

            sent_embs = embed_model.encode(sentences)
            diffs = np.linalg.norm(np.diff(sent_embs, axis=0), axis=1)
            if len(diffs) == 0 or np.isnan(np.mean(diffs)):
                logger.warning(f"Skipping page {page_num}: invalid embedding diffs.")
                continue

            avg_diff = np.mean(diffs)
            base_window = int(len(sentences) * (target_tokens / (target_tokens + avg_diff * 1000)))
            base_window = max(3, min(base_window, len(sentences)))

            i = 0
            while i < len(sentences):
                local_diffs = diffs[i:min(i + base_window, len(diffs))]
                local_var = np.var(local_diffs) if len(local_diffs) else 0
                window = base_window

                if local_var > np.percentile(diffs, 75):
                    window = max(2, base_window // 2)

                start = max(0, i - int(overlap_pct * window))
                end = min(len(sentences), i + window)
                chunk_sents = sentences[start:end]
                chunk_text = ". ".join(chunk_sents).strip()

                if chunk_text:
                    chunks.append({
                        "page": page_num,
                        "text": chunk_text,
                        "size": len(tokenizer.tokenize(chunk_text))
                    })

                i += window - int(overlap_pct * window)

        logger.info(f"✅ Created {len(chunks)} chunks from {pdf_path}")
        return chunks

    except Exception as e:
        logger.exception(f"❌ Error while processing PDF: {e}")
        return []


# def split_on_coherence(sentences, embeddings, sim_threshold=0.7):
#     """
#     Break a list of sentences whenever adjacent similarity < threshold.
#     """
#     boundaries = [0]
#     for i in range(len(embeddings) - 1):
#         sim = cosine_similarity(
#             embeddings[i].reshape(1, -1),
#             embeddings[i+1].reshape(1, -1)
#         )[0,0]
#         if sim < sim_threshold:
#             boundaries.append(i+1)
#     boundaries.append(len(sentences))

#     chunks = []
#     for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
#         chunk = ". ".join(sentences[b_start:b_end]).strip()
#         chunks.append(chunk)
#     return chunks
