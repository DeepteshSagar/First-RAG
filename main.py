"""
main.py
-------
Command-line chat loop for the PDF RAG chatbot.
Run with: python main.py
"""

from src.search import RAGSearch
from src.llm import GeminiAnswerer


def main():
    print("[INFO] Starting PDF RAG chatbot. Loading / building index...")
    rag = RAGSearch()          # builds the FAISS index on first run, loads it after
    answerer = GeminiAnswerer()  # needs GOOGLE_API_KEY in your .env file

    print("\nAsk a question about your documents (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        context = rag.retrieve_as_context(question, top_k=4)
        answer = answerer.answer(question, context)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
