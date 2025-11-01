# app/ui.py
import streamlit as st
from app.retrieval import retrieve
from app.generation import generate_answer
from app.logging import get_logger

logger = get_logger()

st.set_page_config(page_title="RAG - Query", layout="wide")
st.title("RAG — Query")

st.markdown(
    "Ask a question about the pre-built collection. Make sure you've run the ingestion script to build the collection."
)

with st.form(key="query_form", clear_on_submit=False):
    question = st.text_input("Ask a question:", placeholder="", max_chars=1000)
    submitted = st.form_submit_button("Ask")

if not submitted:
    st.info("Type a question above and press Enter or click Ask.")
else:
    q = (question or "").strip()
    if not q:
        st.warning("Please enter a question before asking.")
    else:
        logger.info(f"User asked question: {q!r}")
        with st.spinner("Retrieving relevant documents..."):
            hits = retrieve(q)
        logger.info(f"Retrieved {len(hits)} chunks: {hits}")

        if not hits:
            st.info("No relevant chunks found. Try re-building the index or rephrasing the question.")
        else:
            with st.spinner("Generating answer…"):
                result = generate_answer(q, hits)
            logger.info(f"Generation result: {result}")

            answer = result.get("answer")
            if answer:
                st.subheader("Answer")
                st.markdown(answer)

                with st.expander("Show metadata (confidence, source pages)"):
                    st.write({
                        "confidence_score": result.get("confidence_score"),
                        "source_pages": result.get("source_pages"),
                        "used_chunk_sizes": result.get("used_chunk_sizes"),
                    })
            else:
                st.error("No answer generated. Check logs for details.")
