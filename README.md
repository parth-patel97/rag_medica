# rag_medica
=======
# 📘 Medicare RAG-Based PDF Retrieval System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer user questions based on the official Medicare PDF document. The system uses a dynamic chunking strategy, vector search via FAISS, and a local open-source LLM (e.g., LLaMA 2 via Ollama) to return structured and context-aware answers.

---

## 🚀 Features

- ✅ Dynamic chunking using semantic variance
- ✅ FAISS-based dense retrieval using `sentence-transformers`
- ✅ Open-source LLM integration via [Ollama](https://ollama.com/)
- ✅ FastAPI-powered query endpoint
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
│   ├── main.py               # FastAPI server
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

### 4. Run Ollama with LLaMA2

Ensure Ollama is installed and running locally:

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

## 🚦 Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to access Swagger UI and test the `/query` endpoint.

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
- **Framework**: FastAPI
- **LLM**: LLaMA 2 via Ollama
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **PDF Parser**: PyMuPDF

---

## 📄 Source Document

Used document:

- [Medicare & You 2024 PDF](https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf)

Place the file in the root as: `10050-medicare-and-you.pdf`
