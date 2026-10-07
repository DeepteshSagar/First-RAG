"""
chat_app.py
-----------
Streamlit chat UI that answers questions from whatever is already sitting
in the data/ folder (auto-built/loaded FAISS index). For a UI where the
user uploads files on the spot instead, see app.py.
Run with: streamlit run chat_app.py
"""

import streamlit as st

from src.search import RAGSearch
from src.llm import GeminiAnswerer

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption("Ask questions about the PDFs in your `data/` folder. Fully free stack: FAISS + sentence-transformers + Gemini.")


@st.cache_resource(show_spinner="Loading / building document index...")
def load_pipeline():
    rag = RAGSearch()
    answerer = GeminiAnswerer()
    return rag, answerer


rag, answerer = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            context = rag.retrieve_as_context(question, top_k=4)
            answer = answerer.answer(question, context)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
