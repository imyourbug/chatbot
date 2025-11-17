from __future__ import annotations
import os
from typing import Any, Dict, Optional
from flask import Flask, jsonify, request
from openai import OpenAI
from chatbot_logic import ChatbotIntentResolver
import dotenv

dotenv.load_dotenv()

app = Flask(__name__)
intent_resolver: Optional[ChatbotIntentResolver] = (
    ChatbotIntentResolver(OpenAI()) if os.getenv("OPENAI_API_KEY") else None
)


@app.route("/classify", methods=["POST"])
def classify() -> Any:
    if request.mimetype != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 415

    payload: Dict[str, Any] = request.get_json(force=True)  # type: ignore[assignment]
    text = (payload.get("text") or "").strip()
    image = payload.get("image")

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    if intent_resolver is None:
        return jsonify({"error": "OPENAI_API_KEY is missing"}), 500

    conversation = [{"role": "user", "content": text}]
    action = intent_resolver.determine_action(conversation)

    return jsonify({"action": action.value, "image": image})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

