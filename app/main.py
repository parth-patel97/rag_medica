from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.retrieval import retrieve
from app.generation import generate_answer

class Query(BaseModel):
    question: str

app = FastAPI(title="Medicare RAG API")

@app.post("/query")
def query_medica(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    hits = retrieve(q.question)
    if not hits:
        return {"answer": "", "source_pages": [], "confidence_score": 0.0, "used_chunk_sizes": []}
    result = generate_answer(q.question, hits)
    return result
