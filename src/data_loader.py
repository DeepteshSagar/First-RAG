"""
data_loader.py
---------------
Walks a data directory (or a list of individually uploaded files) and
converts every supported file into LangChain Document objects. Each loader
is just a callable that takes a file path string and returns something with
a `.load()` method - that's what lets us keep this as a simple lookup table
instead of a long if/elif chain, and why adding Excel support below doesn't
need a new branch of logic, just one more dictionary entry.
"""

import tempfile
from pathlib import Path
from typing import List, Iterable, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


def _load_excel(path: str) -> List[Document]:
    """Excel has no first-class LangChain loader in our dependency set, so we
    read it with pandas and turn each sheet into one Document. Each row
    becomes a CSV-style line, which keeps the text readable to both the
    embedding model and the LLM."""
    import pandas as pd

    docs = []
    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue
        docs.append(
            Document(
                page_content=df.to_csv(index=False),
                metadata={"source": path, "sheet": sheet_name},
            )
        )
    return docs


class _CallableLoader:
    """Wraps a plain function in a `.load()` method so it fits the same
    interface as every real LangChain loader (PyPDFLoader, TextLoader, ...).
    This is what lets LOADERS stay a flat dictionary even for formats, like
    Excel, that LangChain doesn't give us a ready-made loader for."""

    def __init__(self, fn, path):
        self._fn = fn
        self._path = path

    def load(self):
        return self._fn(self._path)


# Map file extension -> loader factory.
# Add a new file type by adding one line here.
LOADERS = {
    ".pdf": lambda path: PyPDFLoader(path),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
    ".docx": lambda path: Docx2txtLoader(path),
    ".xlsx": lambda path: _CallableLoader(_load_excel, path),
    ".xls": lambda path: _CallableLoader(_load_excel, path),
}


def load_all_documents(data_dir: str) -> List[Document]:
    """
    Recursively load every supported file under `data_dir` into LangChain
    Document objects.

    Supported today: PDF, TXT, MD, CSV, DOCX, XLSX/XLS.
    """
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Scanning for documents in: {data_path}")

    if not data_path.exists():
        print(f"[WARN] Data directory does not exist: {data_path}")
        return []

    documents: List[Document] = []

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue

        make_loader = LOADERS.get(file_path.suffix.lower())
        if make_loader is None:
            continue  # unsupported extension, skip silently

        try:
            loader = make_loader(str(file_path))
            loaded_docs = loader.load()
            print(f"[INFO] Loaded {len(loaded_docs)} doc(s) from {file_path.name}")
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path.name}: {e}")

    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents


def load_uploaded_files(uploaded_files: Iterable[Any]) -> List[Document]:
    """
    Load documents from Streamlit's `st.file_uploader` output (or any
    objects with `.name` and `.getvalue()` / `.read()`).

    Every loader we have (PyPDFLoader, Docx2txtLoader, etc.) expects a file
    *path* on disk, not raw bytes in memory - so each uploaded file is
    written to a temporary directory first, then handed to the same
    `LOADERS` table `load_all_documents` uses. This keeps upload handling
    and folder handling sharing one piece of loading logic instead of two.
    """
    documents: List[Document] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix.lower()
            make_loader = LOADERS.get(suffix)
            if make_loader is None:
                print(f"[WARN] Unsupported file type, skipping: {uploaded_file.name}")
                continue

            tmp_path = Path(tmp_dir) / uploaded_file.name
            # UploadedFile supports getvalue(); plain file objects support read().
            data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
            tmp_path.write_bytes(data)

            try:
                loader = make_loader(str(tmp_path))
                loaded_docs = loader.load()
                # Replace the temp-dir path in metadata with the original
                # filename so citations shown to the user make sense.
                for d in loaded_docs:
                    d.metadata["source"] = uploaded_file.name
                print(f"[INFO] Loaded {len(loaded_docs)} doc(s) from {uploaded_file.name}")
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"[ERROR] Failed to load {uploaded_file.name}: {e}")

    return documents


if __name__ == "__main__":
    # Quick manual test: python -m src.data_loader
    docs = load_all_documents("data")
    for d in docs[:3]:
        print(d.metadata)
