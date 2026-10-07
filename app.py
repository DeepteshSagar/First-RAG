"""
app.py
------
Streamlit dashboard: upload PDF / Word / CSV / Excel / text files, ask a
question, get an answer grounded in exactly the files you uploaded.

Unlike chat_app.py (which reads from a data/ folder on disk), this
dashboard builds its index entirely in memory for the current session -
nothing is written to disk unless you tick "Save index to disk" below.

Run with: streamlit run app.py
"""

import streamlit as st

from src.search import RAGSearch
from src.llm import GeminiAnswerer

st.set_page_config(page_title="Deeptesh's RAG Dashboard", page_icon="\U0001F4DA", layout="wide")
st.title("\U0001F4DA Deeptesh's RAG Dashboard")
st.caption(
    "Upload PDFs, Word docs, CSVs, Excel sheets, or text files, then ask a question "
    "about them. Fully free stack: FAISS + sentence-transformers (local embeddings) + Gemini."
)

SUPPORTED_TYPES = ["pdf", "docx", "csv", "xlsx", "xls", "txt", "md"]


@st.cache_resource(show_spinner="Starting up (loading embedding model)...")
def get_pipeline():
    # autoload=False: this dashboard only ever answers from files the
    # current user uploads in this session, never from a data/ folder.
    rag = RAGSearch(persist_dir="faiss_store_dashboard", autoload=False)
    answerer = GeminiAnswerer()
    return rag, answerer


rag, answerer = get_pipeline()

# Tracks (filename, size) pairs already embedded, so re-running the script
# on every Streamlit interaction doesn't re-embed the same file repeatedly.
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, Word (.docx), CSV, Excel (.xlsx/.xls), TXT or Markdown",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    persist = st.checkbox(
        "Also save this index to disk (faiss_store_dashboard/)",
        value=False,
        help="Off by default so uploaded files don't linger on disk after the session ends.",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if (f.name, f.size) not in st.session_state.indexed_files
        ]
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                summary = rag.index_uploaded_files(new_files, persist=persist)
            for f in new_files:
                st.session_state.indexed_files.append((f.name, f.size))
            st.success(
                f"Indexed {summary['documents_loaded']} document(s) "
                f"into {summary['chunks_added']} chunks."
            )

    st.divider()
    if st.session_state.indexed_files:
        st.markdown(f"**Indexed files ({len(st.session_state.indexed_files)}):**")
        for name, _ in st.session_state.indexed_files:
            st.markdown(f"- {name}")
    else:
        st.info("No files indexed yet. Upload something above to get started.")

st.header("2. Ask a question")
question = st.text_input("Your question", placeholder="e.g. What is the total revenue in Q3?")
top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)

generate = st.button("Generate answer", type="primary", disabled=not rag.has_documents())

if not rag.has_documents():
    st.info("Upload at least one document in the sidebar to enable the Generate button.")

if generate:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            context = rag.retrieve_as_context(question, top_k=top_k)
            answer = answerer.answer(question, context)

        st.subheader("Answer")
        st.markdown(answer)

        with st.expander("Show retrieved context (what the LLM actually saw)"):
            hits = rag.retrieve(question, top_k=top_k)
            if not hits:
                st.write("No matching chunks were retrieved.")
            for h in hits:
                extra = f", page {h['page']}" if h.get("page") is not None else ""
                extra += f", sheet {h['sheet']}" if h.get("sheet") is not None else ""
                st.markdown(f"**{h['source']}{extra}** · similarity {h['score']:.3f}")
                st.text(h["text"][:500] + ("..." if len(h["text"]) > 500 else ""))
