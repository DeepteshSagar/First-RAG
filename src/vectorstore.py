"""
vectorstore.py
--------------
Stores chunk embeddings in a FAISS index and lets us search them later.
FAISS is free, runs locally, and needs no server - good fit for a fully
free project.
"""

import os
import pickle
from typing import List, Any, Dict, Optional

import faiss
import numpy as np

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        pipeline: Optional[EmbeddingPipeline] = None,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None            # faiss index, created on first add_embeddings call
        self.metadata: List[Dict] = []  # metadata[i] describes the vector at row i

        self.embedding_model = embedding_model
        # One pipeline = one loaded model, reused for both document chunks
        # and later for embedding the user's query. Pass an existing pipeline
        # in (e.g. from a Streamlit cache) to avoid reloading the model every
        # time a new vector store is created in the same session.
        self.pipeline = pipeline or EmbeddingPipeline(embedding_model, chunk_size, chunk_overlap)

    # ---------- building the index ----------

    def add_documents(self, documents: List[Any]) -> int:
        """
        Chunk -> embed -> add, WITHOUT saving. This is the piece that's
        reusable for both a one-shot build (below) and for adding more
        files to an already-open store later, e.g. every time the user
        uploads another batch of files in the Streamlit dashboard.
        Returns the number of chunks added, so callers can show it to the user.
        """
        if not documents:
            return 0

        chunks = self.pipeline.chunk_documents(documents)
        if not chunks:
            return 0

        embeddings = self.pipeline.embed_chunks(chunks)
        metadatas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
                # Excel documents carry a "sheet" name instead of a page
                # number (see data_loader._load_excel) - keep it if present
                # so the UI can show the user exactly where an answer came from.
                "sheet": chunk.metadata.get("sheet"),
            }
            for chunk in chunks
        ]
        self.add_embeddings(embeddings, metadatas)
        return len(chunks)

    def build_from_documents(self, documents: List[Any]) -> None:
        """Chunk -> embed -> add -> save, in one call. Used for the initial
        build from a data/ folder (see search.py / main.py)."""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        n_chunks = self.add_documents(documents)
        self.save()
        print(f"[INFO] Vector store built ({n_chunks} chunks) and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        embeddings = np.asarray(embeddings, dtype="float32")

        # Normalising each vector to unit length lets us use an inner-product
        # index (IndexFlatIP) to compute cosine similarity, which is a more
        # reliable "how similar are these two texts" score than raw distance.
        faiss.normalize_L2(embeddings)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    # ---------- persistence ----------

    def save(self) -> None:
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        self.index = faiss.read_index(os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)  # only unpickle files this project created
        print(f"[INFO] Loaded {self.index.ntotal} vectors from {self.persist_dir}")

    def exists(self) -> bool:
        return os.path.exists(os.path.join(self.persist_dir, "faiss.index")) and os.path.exists(
            os.path.join(self.persist_dir, "metadata.pkl")
        )

    # ---------- search ----------

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        query_embedding = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, ids = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when there are fewer than top_k matches
            results.append({"score": float(score), **self.metadata[idx]})
        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Convenience method: embed the text and search in one call."""
        query_embedding = self.pipeline.embed_text(query_text)
        return self.search(query_embedding, top_k)
