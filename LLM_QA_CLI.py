#!/usr/bin/env python3
"""
LLM_QA_CLI.py

- Provides get_answer(question: str) -> dict:
    Returns {"ok": True/False, "answer": str, "raw": dict, "error": str (if any)}

- When run as a script, provides a small interactive CLI:
    $ python LLM_QA_CLI.py
    > Type your question...
"""

import os
import json
import re
import string
import sys
from typing import Dict, Any, List, Tuple

import requests

# -----------------------
# Configuration (env)
# -----------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq-llama3-13b")
GROQ_OPENAI_BASE = os.environ.get("GROQ_OPENAI_BASE", "https://api.groq.com/openai/v1")
CHAT_COMPLETIONS_URL = f"{GROQ_OPENAI_BASE}/chat/completions"

# Note: we intentionally do not raise at import time if the API key is missing to allow
# import-time usage for things like documentation generation. get_answer() will check.
# -----------------------
# Utilities (preprocessing)
# -----------------------
def simple_preprocess(text: str) -> Tuple[str, List[str]]:
    """
    Basic preprocessing:
    - lowercasing
    - remove punctuation
    - simple whitespace tokenization
    Returns: (cleaned_text, tokens)
    """
    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    cleaned = text.lower()

    # Remove punctuation (preserves basic inner apostrophes if desired; here we remove all)
    translator = str.maketrans("", "", string.punctuation)
    cleaned = cleaned.translate(translator)

    # Collapse multiple whitespace and trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Tokenize on space (very simple tokenizer)
    tokens = cleaned.split(" ") if cleaned else []

    return cleaned, tokens

# -----------------------
# LLM call (requests -> plain dict)
# -----------------------
def _call_groq_chat_raw(question: str, model: str = GROQ_MODEL, timeout: int = 30) -> Dict[str, Any]:
    """
    Make the raw HTTP request to Groq's OpenAI-compatible chat completion endpoint.
    Returns the parsed JSON as a plain Python dict (raw provider response).
    Raises requests.exceptions.HTTPError on non-2xx.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    resp = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# -----------------------
# Public function: get_answer
# -----------------------
def get_answer(question: str, model: str = GROQ_MODEL, timeout: int = 30) -> Dict[str, Any]:
    """
    High-level function used by the Flask app and the CLI.

    Returns a plain dict:
      {
        "ok": True/False,
        "answer": "<string>",
        "raw": {<raw provider json>} or None,
        "error": "<error message>" or None
      }
    """
    try:
        if not question or not str(question).strip():
            return {"ok": False, "answer": "", "raw": None, "error": "Empty question."}

        # Basic preprocessing
        cleaned, tokens = simple_preprocess(question)

        # Construct a prompt — here we keep the original question but you might include
        # the cleaned version or token stats in a system prompt if desired.
        # For example we will add a short instruction to the system role to keep responses concise.
        prompt = question  # we send the original question to the model (more natural)

        raw = _call_groq_chat_raw(prompt, model=model, timeout=timeout)

        # Defensive extract assistant text (OpenAI-compatible response shape)
        assistant_text = ""
        try:
            choices = raw.get("choices", [])
            if choices and isinstance(choices, list):
                first = choices[0] or {}
                # Support both 'message' or 'text' legacy shapes:
                if "message" in first and isinstance(first["message"], dict):
                    assistant_text = first["message"].get("content", "") or ""
                elif "text" in first:
                    assistant_text = first.get("text", "") or ""
                else:
                    # fallback: try join on choice.delta or other nested fields
                    assistant_text = str(first)
        except Exception:
            assistant_text = ""

        return {"ok": True, "answer": assistant_text, "raw": raw, "error": None, "preprocessed": {"cleaned": cleaned, "tokens": tokens}}

    except requests.exceptions.HTTPError as he:
        # provider returned a non-2xx
        try:
            body = he.response.json()
        except Exception:
            body = he.response.text if he.response is not None else str(he)
        return {"ok": False, "answer": "", "raw": body, "error": f"Provider HTTP error: {str(he)}"}

    except Exception as e:
        # Generic error
        return {"ok": False, "answer": "", "raw": None, "error": str(e)}


# -----------------------
# CLI entrypoint
# -----------------------
def _interactive_cli():
    """
    Simple interactive CLI loop. Type 'quit' or Ctrl+C to exit.
    """
    print("LLM Q&A CLI — ask a natural-language question (type 'quit' to exit)")
    try:
        while True:
            question = input("\n> ").strip()
            if not question:
                print("Please type a non-empty question (or 'quit' to exit).")
                continue
            if question.lower() in ("quit", "exit"):
                print("Goodbye.")
                break

            result = get_answer(question)
            if result.get("ok"):
                print("\nAnswer:\n")
                print(result.get("answer", "(no answer)"))
                # Small helpful info for novices
                meta = result.get("preprocessed")
                if meta:
                    tokens = meta.get("tokens", [])
                    print(f"\n[preprocessing: {len(tokens)} tokens]")
            else:
                print("\nError:", result.get("error"))
                raw = result.get("raw")
                if raw:
                    print("\nRaw provider response (truncated):")
                    try:
                        # pretty print a tiny slice
                        print(json.dumps(raw if isinstance(raw, dict) else {"raw": str(raw)}, indent=2)[:2000])
                    except Exception:
                        print(str(raw)[:2000])
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    # When executed directly, run CLI
    _interactive_cli()