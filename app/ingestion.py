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
    """
    chunks = []

    try:
        doc = fitz.open(pdf_path)
        # try to use tokenizer if present, fallback to a simple heuristic
        tokenizer = getattr(embed_model, "tokenizer", None)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text:
                continue

            sentences = [s.strip() for s in text.split(". ") if s.strip()]
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
                    # compute token size if tokenizer available, else approximate by words
                    try:
                        if tokenizer:
                            size = len(tokenizer.tokenize(chunk_text))
                        else:
                            size = len(chunk_text.split())
                    except Exception:
                        size = len(chunk_text.split())

                    chunks.append({
                        "page": page_num,
                        "text": chunk_text,
                        "size": int(size)
                    })

                i += max(1, window - int(overlap_pct * window))

        logger.info(f"✅ Created {len(chunks)} chunks from {pdf_path}")
        return chunks

    except Exception as e:
        logger.exception(f"❌ Error while processing PDF: {e}")
        return []
