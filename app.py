# app.py
import os
import json
from flask import Flask, render_template, request, jsonify
import requests

# -------------------------
# Configuration (env vars)
# -------------------------
# Required: set these in your environment (Render dashboard or locally)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Model name is environment-configurable so you can change it without changing code.
# IMPORTANT: confirm the exact model name available in your Groq account/console.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq-llama3-13b")  # change if needed

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set. See README or Render env settings.")

# Use the OpenAI-compatible Groq base URL (the Groq docs expose an OpenAI-compatible endpoint)
GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"
CHAT_COMPLETIONS_URL = f"{GROQ_OPENAI_BASE}/chat/completions"

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------------
# Helpers
# -------------------------
def call_groq_chat(question: str, model: str = GROQ_MODEL, timeout: int = 30) -> dict:
    """
    Calls Groq's chat completions endpoint (OpenAI-compatible endpoint).
    Returns a plain Python dict with the provider response.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        # tune these if needed:
        "temperature": 0.2,
        "max_tokens": 512
    }

    resp = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()  # raises HTTPError for 4xx/5xx

    # Convert response JSON into a plain dict
    data = resp.json()

    # Defensive parsing to extract the assistant text:
    assistant_text = ""
    try:
        # OpenAI-style response structure: choices[0].message.content
        assistant_text = data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        assistant_text = ""

    return {"raw": data, "answer": assistant_text}

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Serves the user UI (we'll create templates/index.html in the next step)
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Expects JSON body: { "question": "<user question here>" }
    Returns JSON: { "answer": "<LLM answer>", "ok": true } on success
    """
    body = request.get_json(force=True, silent=True)
    if not body or "question" not in body:
        return jsonify({"ok": False, "error": "Missing 'question' in request body."}), 400

    question = body["question"]
    try:
        groq_resp = call_groq_chat(question)
        # Return only plain structures (strings/dicts/lists)
        return jsonify({"ok": True, "answer": groq_resp.get("answer", ""), "meta": {"model": GROQ_MODEL}})
    except requests.exceptions.RequestException as e:
        # network error or non-2xx from provider
        return jsonify({"ok": False, "error": "API request failed", "details": str(e)}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": "Internal server error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "healthy"})

# -------------------------
# Run (development)
# -------------------------
if __name__ == "__main__":
    # For local dev: FLASK_ENV=development python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
