from pathlib import Path
import pickle
from urllib.parse import urlparse

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = Flask(__name__)
CORS(app)

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


TRUSTED_SOURCES = {
    "reuters",
    "associated press",
    "ap news",
    "bbc",
    "bbc news",
    "the hindu",
    "indian express",
    "times of india",
    "hindustan times",
    "ndtv",
    "the guardian",
    "nasa",
    "who",
    "unicef",
    "imd",
    "government",
    "ministry",
}

TRUSTED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "thehindu.com",
    "indianexpress.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "ndtv.com",
    "nasa.gov",
    "who.int",
    "gov.in",
}

SUSPICIOUS_PATTERNS = {
    "doctors hate",
    "one trick",
    "click now",
    "secret cure",
    "guarantees",
    "miracle",
    "anonymous post",
    "viral message",
    "forward this",
    "shocking",
    "hidden truth",
    "tracking chips",
    "moonlight water",
}


def get_domain(url: str) -> str:
    if not url:
        return ""

    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def source_is_trusted(source: str, domain: str) -> bool:
    source_value = (source or "").strip().lower()
    if source_value in TRUSTED_SOURCES:
        return True

    return any(domain == trusted or domain.endswith(f".{trusted}") for trusted in TRUSTED_DOMAINS)


def suspicious_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for pattern in SUSPICIOUS_PATTERNS if pattern in lowered)


@app.get("/")
def home():
    return jsonify({"message": "Smart News Verifier ML service is running."})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    source = data.get("source", "").strip()
    url = data.get("url", "").strip()

    if not text:
        return jsonify({"message": "Text is required."}), 400

    vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(vectorized)[0]
    base_true_prob = float(probabilities[1])
    domain = get_domain(url)
    trusted = source_is_trusted(source, domain)
    suspicious_hits = suspicious_score(text)

    adjusted_true_prob = base_true_prob
    if trusted:
        adjusted_true_prob += 0.18
    if suspicious_hits:
        adjusted_true_prob -= min(0.12 * suspicious_hits, 0.36)

    adjusted_true_prob = max(0.02, min(0.98, adjusted_true_prob))

    label = "TRUE" if adjusted_true_prob >= 0.5 else "FAKE"
    confidence = round(
        (adjusted_true_prob if label == "TRUE" else (1 - adjusted_true_prob)) * 100,
        2
    )

    return jsonify({
        "label": label,
        "confidence": confidence,
        "trusted_source": trusted
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
