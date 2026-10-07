"""
search.py
---------
Ties data_loader and vectorstore together. On first run it builds the
FAISS index from the PDFs/text files in `data/`; on later runs it just
loads the saved index from disk, so you don't re-embed every time.
"""

from typing import List, Dict, Iterable, Any

from src.data_loader import load_all_documents, load_uploaded_files
from src.vectorstore import FaissVectorStore


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        data_dir: str = "data",
        embedding_model: str = "all-MiniLM-L6-v2",
        autoload: bool = True,
    ):
        """
        autoload=True (used by main.py / app.py's data/-folder flow): on
        first run, build the index from every file already sitting in
        `data_dir`; on later runs, load the saved index instead.

        autoload=False (used by the upload dashboard): start with an empty,
        unsaved store. Documents only enter the index when
        `index_uploaded_files` is called, so the dashboard only ever answers
        from files the current user actually uploaded.
        """
        self.store = FaissVectorStore(persist_dir, embedding_model)

        if not autoload:
            return

        if self.store.exists():
            self.store.load()
        else:
            docs = load_all_documents(data_dir)
            if not docs:
                print(
                    f"[WARN] No documents found in '{data_dir}'. "
                    "Add PDFs/TXT/CSV files there and re-run."
                )
            self.store.build_from_documents(docs)

    def has_documents(self) -> bool:
        return self.store.index is not None and self.store.index.ntotal > 0

    def index_uploaded_files(self, uploaded_files: Iterable[Any], persist: bool = False) -> Dict:
        """
        Load a batch of Streamlit-uploaded files (PDF/DOCX/CSV/XLSX/TXT/MD),
        chunk and embed them, and add them to the store in memory.

        persist=False keeps the dashboard's index session-only, which is the
        right default since uploads can contain things the user doesn't want
        written to disk. Pass persist=True to also save to `self.store.persist_dir`.

        Returns a small summary dict so the UI can tell the user what happened.
        """
        uploaded_files = list(uploaded_files)  # materialise once; avoids consuming a generator twice
        docs = load_uploaded_files(uploaded_files)
        n_chunks = self.store.add_documents(docs)
        if persist:
            self.store.save()

        return {
            "files_received": len(uploaded_files),
            "documents_loaded": len(docs),
            "chunks_added": n_chunks,
        }

    def retrieve(self, question: str, top_k: int = 4) -> List[Dict]:
        """Return the top_k most relevant chunks for a question, each with
        its similarity score, source file, page number and text."""
        return self.store.query(question, top_k)

    def retrieve_as_context(self, question: str, top_k: int = 4) -> str:
        """Format retrieved chunks into one context string for the LLM prompt,
        tagging each chunk with its source so answers can cite where they came from."""
        hits = self.retrieve(question, top_k)
        if not hits:
            return ""

        parts = []
        for h in hits:
            extra = f", page {h['page']}" if h.get("page") is not None else ""
            extra += f", sheet {h['sheet']}" if h.get("sheet") is not None else ""
            parts.append(f"[Source: {h['source']}{extra}]\n{h['text']}")
        return "\n\n---\n\n".join(parts)
