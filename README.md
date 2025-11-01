# rag_medica

# 📘 Medicare RAG-Based PDF Retrieval System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer user questions based on the official Medicare PDF document. The system uses a dynamic chunking strategy, vector search via FAISS, and a local open-source LLM (e.g., LLaMA 2 via Ollama) to return structured and context-aware answers.

---

## 🚀 Features

- ✅ Dynamic chunking using semantic variance
- ✅ FAISS-based dense retrieval using `sentence-transformers`
- ✅ Open-source LLM integration via [Ollama](https://ollama.com/)
- ✅ Query endpoint
- ✅ Structured JSON output with source tracking

---

## 🧠 Example Query

**Input:**

```json
{
  "question": "What are the important deadlines for Medicare enrollment?"
}
```

**Output:**

```json
{
  "answer": "Medicare enrollment begins on October 15 and ends on December 7 each year...",
  "source_pages": [15],
  "confidence_score": 0.92,
  "used_chunk_sizes": [230, 210, 198]
}
```

---

## 📂 Project Structure

```
rag_medica/
│
├── app/
│   ├── main.py               # server entrypoint
│   ├── ingestion.py          # PDF chunking with semantic logic
│   ├── retrieval.py          # FAISS-based top-k retriever
│   ├── embeddings.py         # Index builder using SentenceTransformer
│   ├── generation.py         # Local LLM query & structured output
│   ├── logging.py            # Centralized logger utility
│   ├── build_index.py        # Script to build the FAISS index
│
├── models/
│   ├── faiss.idx             # FAISS index
│   ├── meta.pkl              # Chunk metadata

```

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/rag-medicare.git
cd rag-medicare
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment variables (Google Gemini / local LLM)

The project can use either a local LLM (via Ollama) or Google Gemini (via the Generative AI API). Set these environment variables before running the app.

- `GEMINI_API_KEY` — (optional) API key for Google Gemini / Generative AI. The code reads `GEMINI_API_KEY` at runtime. If you prefer a different name, update `app/generation.py` accordingly.
- `GOOGLE_GENAI_MODEL` — (optional) model name to use (default: `gemini-2.5-flash`).
- `GOOGLE_GENAI_TIMEOUT` — (optional) request timeout in seconds (default: `60`).

Set environment variables on macOS / Linux like:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GOOGLE_GENAI_MODEL="gemini-2.5-flash"
export GOOGLE_GENAI_TIMEOUT=60
```

If you do not set `GEMINI_API_KEY`, the app will attempt to use the local LLM flow (Ollama) if available.

### 5. Run Ollama with LLaMA2 (optional local LLM)

Ensure Ollama is installed and running locally if you want to use a local model instead of Google Gemini:

```bash
ollama run llama2
```

---

## 🧱 Build Index

Before querying, you must create chunks and build the index:

```bash
python app/build_index.py
```

---

## 🚦 Start the server

Check `app/main.py` for the project's server entrypoint and start instructions. (This repo no longer prescribes a specific ASGI server.)

---

## 📤 API Endpoint

### `POST /query`

**Request Body:**

```json
{
  "question": "Your query here"
}
```

**Response:**
Returns:

- `answer`: Generated response from LLM
- `source_pages`: Pages from the PDF used
- `confidence_score`: Average cosine score from FAISS
- `used_chunk_sizes`: Token size of the chunks used

---

## 🧪 Edge Case Handling

- Empty queries are rejected with `400 Bad Request`.
- If no relevant chunks are found, returns empty fields with `0.0` confidence.

---

## 🛠 Tech Stack

- **Language**: Python
- **LLM**: LLaMA 2 via Ollama
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **PDF Parser**: PyMuPDF

---

## 📄 Source Document

Used document:

- [Medicare & You 2024 PDF](https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf)

Place the file in the root as: `10050-medicare-and-you.pdf`
