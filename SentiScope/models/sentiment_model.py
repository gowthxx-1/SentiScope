"""
models/sentiment_model.py
Multi-model sentiment analysis engine:
1. VADER — rule-based, fast, great for social-media slang & emojis
2. TextBlob — pattern-based, lightweight baseline
3. Transformer — DistilBERT / RoBERTa fine-tuned on tweets (most accurate)
4. Ensemble — weighted vote: Transformer 50% + VADER 30% + TextBlob 20%
"""

import re
import nltk
import torch
import numpy as np
from typing import Dict, List, Optional
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from loguru import logger

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Text Preprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = self._remove_urls(text)
        text = self._remove_mentions_hashtags(text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def clean_for_vader(self, text: str) -> str:
        text = self._remove_urls(text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"RT\s+", "", text)
        return text.strip()

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    def _remove_mentions_hashtags(self, text: str) -> str:
        text = re.sub(r"@\w+", "", text)
        return re.sub(r"#(\w+)", r"\1", text)


# ── VADER ─────────────────────────────────────────────────────────────────────
class VADERAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.preprocessor = TextPreprocessor()

    def analyze(self, text: str) -> dict:
        cleaned = self.preprocessor.clean_for_vader(text)
        scores = self.analyzer.polarity_scores(cleaned)
        compound = scores["compound"]

        if compound >= 0.05:
            label, confidence = "positive", 0.5 + compound * 0.5
        elif compound <= -0.05:
            label, confidence = "negative", 0.5 + abs(compound) * 0.5
        else:
            label, confidence = "neutral", 1 - abs(compound)

        return {
            "model": "vader",
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "compound": round(compound, 4),
            },
        }


# ── TextBlob ──────────────────────────────────────────────────────────────────
class TextBlobAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def analyze(self, text: str) -> dict:
        cleaned = self.preprocessor.clean(text)
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            label, confidence = "positive", 0.5 + polarity * 0.5
        elif polarity < -0.1:
            label, confidence = "negative", 0.5 + abs(polarity) * 0.5
        else:
            label, confidence = "neutral", 1 - abs(polarity)

        return {
            "model": "textblob",
            "label": label,
            "confidence": round(confidence, 4),
            "scores": {
                "polarity": round(polarity, 4),
                "subjectivity": round(subjectivity, 4),
            },
        }


# ── Transformer (lazy-loaded) ─────────────────────────────────────────────────
class TransformerAnalyzer:
    PRIMARY_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self._pipeline = None  # lazy

    def _load(self):
        if self._pipeline is not None:
            return
        device = 0 if torch.cuda.is_available() else -1
        for model_id in (self.PRIMARY_MODEL, self.FALLBACK_MODEL):
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_id,
                    device=device,
                    truncation=True,
                    max_length=512,
                )
                logger.info(f"Loaded transformer: {model_id}")
                return
            except Exception as exc:
                logger.warning(f"Could not load {model_id}: {exc}")
        raise RuntimeError("No transformer model could be loaded.")

    def analyze(self, text: str) -> dict:
        self._load()
        cleaned = self.preprocessor.clean_for_vader(text)
        result = self._pipeline(cleaned[:512])[0]
        raw = result["label"].lower()
        if "pos" in raw or raw == "label_2":
            label = "positive"
        elif "neg" in raw or raw == "label_0":
            label = "negative"
        else:
            label = "neutral"
        return {
            "model": "transformer",
            "label": label,
            "confidence": round(result["score"], 4),
        }


# ── Ensemble ──────────────────────────────────────────────────────────────────
class EnsembleSentimentAnalyzer:
    """Weighted majority vote: Transformer 50% + VADER 30% + TextBlob 20%."""

    WEIGHTS: Dict[str, float] = {
        "transformer": 0.50,
        "vader": 0.30,
        "textblob": 0.20,
    }

    def __init__(self, use_transformer: bool = True):
        self.vader = VADERAnalyzer()
        self.textblob = TextBlobAnalyzer()
        self.transformer = TransformerAnalyzer() if use_transformer else None

    def analyze(self, text: str) -> dict:
        results: Dict[str, dict] = {}
        results["vader"] = self.vader.analyze(text)
        results["textblob"] = self.textblob.analyze(text)
        if self.transformer:
            try:
                results["transformer"] = self.transformer.analyze(text)
            except Exception as exc:
                logger.error(f"Transformer failed: {exc}")

        label_scores: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_weight = 0.0
        for model_name, result in results.items():
            w = self.WEIGHTS.get(model_name, 0.1)
            label_scores[result["label"]] += w * result["confidence"]
            total_weight += w

        for k in label_scores:
            label_scores[k] /= total_weight if total_weight else 1

        final_label = max(label_scores, key=lambda k: label_scores[k])
        final_confidence = label_scores[final_label]

        return {
            "text": text[:280],
            "sentiment": final_label,
            "confidence": round(final_confidence, 4),
            "label_scores": {k: round(v, 4) for k, v in label_scores.items()},
            "model_results": results,
            "emoji": {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(final_label, "🤔"),
        }

    def batch_analyze(self, texts: List[str]) -> List[dict]:
        return [self.analyze(t) for t in texts]
   
