import requests
import json
import numpy as np
from app.logging import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
You are an expert on Medicare.

Question:
{question}

Relevant Context Chunks:
{chunks}

Return JSON with keys:
- answer: (string)
- source_pages: (list of integers)
- confidence_score: (float from 0 to 1)
- used_chunk_sizes: (list of integers)
"""

def generate_answer(question, hits):
    """
    Calls the local LLM via API to generate an answer based on retrieved chunks.

    Args:
        question (str): User query.
        hits (List[dict]): Retrieved chunks with metadata.

    Returns:
        dict: Structured JSON with answer, source pages, confidence, and chunk sizes.
    """
    try:
        chunks_text = "\n\n".join(
            f"--- Page {h['page']} ---\n{h['text']}" for h in hits
        )
        prompt = PROMPT_TEMPLATE.format(question=question, chunks=chunks_text)

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt, "stream": False},
        )
        output_text = response.json()["response"]
        raw_answer = json.loads(output_text)["answer"]

        confidence_score = float(np.mean([h["score"] for h in hits]))
        source_pages = sorted(list(set([h["page"] for h in hits])))
        chunk_sizes = [h["size"] for h in hits]

        logger.info(f"✅ Answer generated for query: {question}")
        return {
            "answer": raw_answer,
            "source_pages": source_pages,
            "confidence_score": round(confidence_score, 3),
            "used_chunk_sizes": chunk_sizes
        }

    except Exception as e:
        logger.exception(f"❌ Failed to generate answer for query: {question}")
        return {
            "answer": "",
            "source_pages": [],
            "confidence_score": 0.0,
            "used_chunk_sizes": []
        }
