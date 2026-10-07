# PDF RAG Chatbot

A fully free Retrieval-Augmented Generation (RAG) chatbot that answers
questions about your own PDF/TXT/CSV documents.

**Stack:** LangChain + sentence-transformers (local embeddings, no API cost) +
FAISS (local vector store, no server) + Gemini via `langchain-google-genai`
(free-tier LLM).

## Project structure

```
.
├── data/                  # put your PDFs / Word / CSV / Excel / text files here
├── src/
│   ├── data_loader.py     # loads files (folder or uploaded) into LangChain Documents
│   ├── embedding.py       # chunks text and creates embeddings
│   ├── vectorstore.py     # stores/searches embeddings with FAISS
│   ├── search.py          # ties loader + vectorstore into one retriever
│   └── llm.py             # sends retrieved context + question to Gemini
├── main.py                # command-line chat loop (reads from data/)
├── chat_app.py            # Streamlit chat UI (reads from data/)
├── app.py                 # Streamlit DASHBOARD: upload files + ask a question
└── requirements.txt
```

Supported file types everywhere in the project: **PDF, DOCX (Word), CSV,
XLSX/XLS (Excel), TXT, MD**.

## Two ways to use it

- **`data/` folder flow** (`main.py`, `chat_app.py`) - drop files into
  `data/` once, the index is built on first run and reused after that.
- **Upload dashboard** (`app.py`) - upload files directly in the browser
  each session; nothing is read from or written to `data/` unless you tick
  "save index to disk".

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your free Gemini API key
   (from https://aistudio.google.com/apikey):
   ```
   GOOGLE_API_KEY=your_key_here
   ```

3. Put your PDF/TXT/CSV files in `data/`.

## Run

Command line (reads `data/`):
```bash
python main.py
```

Chat UI (reads `data/`):
```bash
streamlit run chat_app.py
```

Upload dashboard (upload files yourself, ask a question, get an answer with
sources — this is the one to use if you don't want to manage a `data/`
folder at all):
```bash
streamlit run app.py
```

On first run, the app builds a FAISS index from everything in `data/` and
saves it to `faiss_store/`. Later runs load that saved index instead of
re-embedding, which is why you won't see the "Building vector store..."
messages every time. Delete `faiss_store/` to force a rebuild after adding
new documents.

## How it works

1. **Load** - `data_loader.py` reads each file and converts it into
   LangChain `Document` objects.
2. **Chunk + Embed** - `embedding.py` splits documents into overlapping
   ~1000-character chunks and turns each chunk into a vector using the
   local `all-MiniLM-L6-v2` model.
3. **Store** - `vectorstore.py` keeps those vectors in a FAISS index on
   disk, alongside the matching chunk text and source metadata.
4. **Retrieve** - `search.py` embeds the user's question with the same
   model and asks FAISS for the most similar chunks.
5. **Generate** - `llm.py` passes those chunks to Gemini as context and
   asks it to answer using only that context.

## How the upload dashboard (app.py) works

1. You upload one or more files in the sidebar (PDF, DOCX, CSV, XLSX/XLS,
   TXT, MD).
2. `data_loader.load_uploaded_files()` writes each file to a temporary
   directory (loaders need a real file path, not just bytes in memory),
   loads it with the same per-extension loader table used for the `data/`
   folder, then deletes the temp copy.
3. `vectorstore.add_documents()` chunks and embeds just that batch and adds
   it to the FAISS index already open in memory - existing files stay
   indexed, so you can keep adding more without starting over.
4. You type a question; the same retrieve -> Gemini flow as `chat_app.py`
   runs, and the answer plus the exact chunks used are shown on screen.

By default the dashboard's index lives only in memory for the session
(`persist=False`). Tick "save index to disk" in the sidebar if you want it
written to `faiss_store_dashboard/` so it survives a restart.

## Possible next steps

- Swap FAISS for Typesense or Chroma by writing a class with the same
  `add_embeddings` / `search` interface as `FaissVectorStore`.
- Add a "sources" expander in the Streamlit UI showing which chunks were
  used for each answer.
- Add conversation memory so follow-up questions can reference earlier turns.
