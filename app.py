import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import pandas as pd
from flask import Flask, jsonify, render_template, request
from openai import OpenAI


data_dir = Path(__file__).parent / "data"


def load_faq_frames(data_path: Path) -> pd.DataFrame:
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    frames = []
    for file in csv_files:
        frame = pd.read_csv(file)
        frame["__source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_context(question: str, faq_frame: pd.DataFrame, limit: int = 3) -> List[str]:
    def similarity(row_text: str) -> float:
        return SequenceMatcher(None, question.lower(), row_text.lower()).ratio()

    faq_frame = faq_frame.copy()
    faq_frame["__text"] = faq_frame.apply(
        lambda row: f"Q: {row.get('question', '')}\nA: {row.get('answer', '')}\nCategory: {row.get('category', '')}",
        axis=1,
    )
    faq_frame["__score"] = faq_frame["__text"].apply(similarity)
    top_rows = faq_frame.nlargest(limit, "__score")
    return top_rows["__text"].tolist()


def create_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


faq_frame = load_faq_frames(data_dir)
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    try:
        client = create_client()
    except EnvironmentError as exc:
        return jsonify({"error": str(exc)}), 500

    context_snippets = build_context(user_message, faq_frame)
    context_text = "\n\n".join(context_snippets)

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful support assistant. Answer using only the information from the provided CSV context. "
                "If the context is insufficient, say you do not have enough information. Keep responses concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context from CSV files:\n{context_text}\n\n"
                f"User question: {user_message}"
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_messages,
            temperature=0.3,
        )
        reply = completion.choices[0].message.content
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to contact OpenAI API: {exc}"}), 500

    return jsonify({"reply": reply, "context": context_snippets})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
