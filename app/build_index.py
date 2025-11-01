# build_index.py
import argparse
import os
from app.ingestion import dynamic_chunk_text_with_variance
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document # adjust if needed
from app.logging import get_logger

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF and build a Chroma collection via LangChain."
    )
    parser.add_argument("--pdf", "-p", required=True, help="Path to the PDF file to ingest")
    parser.add_argument(
        "--persist-dir", default="models/chroma", help="Chroma persist directory"
    )
    parser.add_argument(
        "--collection", default="collection_sample", help="Chroma collection name"
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found: {pdf_path}")
        return

    try:
        logger.info(f"📥 Creating chunks from: {pdf_path}")
        chunks = dynamic_chunk_text_with_variance(pdf_path)
        if not chunks:
            logger.error("❌ No chunks were created from the document. Aborting indexing.")
            return

        persist_dir = os.path.abspath(args.persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        logger.info(f"📦 Building Chroma collection '{args.collection}' at '{persist_dir}'")

        # prepare embeddings
        embedding_model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # prepare documents (adjust Document import if different)
        docs = []
        for i, c in enumerate(chunks):
            docs.append(
                Document(page_content=c["text"], metadata={"page": c.get("page"), "size": c.get("size"), "id": i})
            )

        # build vector store
        vect = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=args.collection
        )

        logger.info("✅ Indexing complete.")

    except Exception as e:
        logger.exception("❌ Index building failed.")

if __name__ == "__main__":
    main()
