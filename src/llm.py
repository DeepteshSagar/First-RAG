"""
llm.py
------
Sends the retrieved context + the user's question to Gemini and returns a
grounded answer. This is the "Generation" step of Retrieval-Augmented
Generation - everything in search.py is the "Retrieval" step.

This talks to Gemini directly over Gemini's REST API using `requests`,
instead of going through the langchain-google-genai / google-genai SDK.
That SDK uses a different transport (HTTP/2 / streaming-style connections)
that got silently blackholed on some networks even when plain HTTPS
requests worked fine and fast - using `requests` sidesteps that entirely.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from a local .env file

# Status codes worth retrying automatically: both mean "try again shortly",
# not "something is wrong with your request".
#   429 = rate limited (too many requests)
#   503 = model temporarily overloaded on Google's side
RETRYABLE_STATUS_CODES = {429, 503}

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below, which was retrieved from the user's own PDF "
    "documents. If the answer is not contained in the context, say you "
    "don't know rather than guessing. Keep answers concise and, where "
    "useful, mention which source the information came from."
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAnswerer:
    def __init__(
        self,
        model_name: str = "gemini-3.7-flash",
        temperature: float = 0.2,
        timeout: int = 60,
        max_retries: int = 2,
        retry_backoff_seconds: float = 3.0,
    ):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Create a .env file with "
                "GOOGLE_API_KEY=your_key (see .env.example)."
            )
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "I couldn't find anything relevant in the uploaded documents."

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        url = f"{API_BASE}/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

        last_error_text = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                # Network-level failure (timeout, connection error) - worth
                # retrying too, since these are often transient blips.
                last_error_text = f"Could not reach Gemini: {e}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    try:
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError):
                        finish_reason = data.get("candidates", [{}])[0].get("finishReason", "unknown")
                        return f"Gemini returned no answer (finish reason: {finish_reason})."

                last_error_text = f"Gemini API error {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return last_error_text  # a real problem (bad key, bad request) - no point retrying

            if attempt < self.max_retries:
                # Exponential backoff: 2s, 4s, 8s... gives a temporarily
                # overloaded model or a rate limit window time to clear.
                wait = self.retry_backoff_seconds * (2 ** (attempt - 1))
                print(f"[WARN] Attempt {attempt} failed, retrying in {wait:.0f}s: {last_error_text}")
                time.sleep(wait)

        return f"Gemini still unavailable after {self.max_retries} attempts. Last error: {last_error_text}"
