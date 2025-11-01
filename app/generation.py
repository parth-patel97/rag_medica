# app/generation.py
import os
import json
from typing import List
from app.logging import get_logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

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

_MODEL = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
_TIMEOUT = int(os.environ.get("GOOGLE_GENAI_TIMEOUT", "60"))  # seconds
_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    if not _API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY environment variable")

    llm = ChatGoogleGenerativeAI(
        model=_MODEL,
        temperature=0,
        max_tokens=None,
        timeout=_TIMEOUT,
        google_api_key=_API_KEY
    )
    logger.info(f"Google Gemini LLM initialized, model={_MODEL}")
except Exception:
    logger.exception("Failed to initialize Google Gemini LLM; llm set to None")
    llm = None

def generate_answer(question: str, hits: List[dict]) -> dict:
    """
    Generate an answer using Google Gemini via LangChain, based on retrieved context chunks.
    """
    try:
        chunks_text = "\n\n".join(f"--- Page {h.get('page')} ---\n{h.get('text')}" for h in hits)
        prompt_text = PROMPT_TEMPLATE.format(question=question, chunks=chunks_text)

        if llm is None:
            raise RuntimeError("LLM not initialized")

        messages = [
            SystemMessage(content="You are an expert on Medicare. Answer the human question based only on the provided context."),
            HumanMessage(content=prompt_text)
        ]

        logger.info("Calling Google Gemini invoke …")
        resp = llm.invoke(messages)
        logger.info("Google Gemini invoke returned")

        content = resp.content if hasattr(resp, "content") else str(resp)
        logger.info("Received output snippet: %s", content[:200])

        try:
            parsed = json.loads(content)
        except Exception:
            try:
                start = content.index("{")
                end = content.rindex("}")
                parsed = json.loads(content[start:end+1])
            except Exception:
                parsed = {"answer": content}

        raw_answer = parsed.get("answer", parsed if isinstance(parsed, str) else "")

        scores = [h.get("score", 0.0) for h in hits] if hits else []
        confidence_score = float(sum(scores) / len(scores)) if scores else 0.0
        source_pages = sorted(list({h.get("page") for h in hits if h.get("page") is not None}))
        chunk_sizes = [h.get("size") for h in hits]

        logger.info(f"✅ Answer generated for query: {question}")
        return {
            "answer": raw_answer,
            "source_pages": source_pages,
            "confidence_score": round(confidence_score, 3),
            "used_chunk_sizes": chunk_sizes,
        }

    except Exception:
        logger.exception(f"❌ Failed to generate answer for query: {question}")
        return {
            "answer": "",
            "source_pages": [],
            "confidence_score": 0.0,
            "used_chunk_sizes": [],
        }
