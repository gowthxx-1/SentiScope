# SentimentPulse 

**Real-Time Sentiment Analysis on Social Media Data**

VADER · TextBlob · DistilBERT | FastAPI · WebSocket · SQLite | React 18 · Netlify

\---

## &#x20;Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\\Scripts\\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file
cp .env.example .env

# 4. Start the backend
uvicorn backend.main:app --reload --port 8000

# 5. Open the dashboard
# Open frontend/dashboard.html in your browser
```

\---

## Project Structure

```
sentimentpulse/
├── backend/
│   ├── main.py           # FastAPI app, WebSocket broadcaster, REST endpoints
│   ├── data\_collector.py # Reddit, HackerNews, RSS collectors (no API keys!)
│   └── database.py       # SQLAlchemy ORM, SQLite, stats queries
├── models/
│   └── sentiment\_model.py # VADER + TextBlob + DistilBERT ensemble
├── frontend/
│   └── dashboard.html    # Single-file React 18 dashboard
├── requirements.txt
├── .env.example
└── README.md
```

\---

## How It Works:

### 

### Data Collection

3 parallel threads pull from **free public APIs** — no API keys required:

* **Reddit** — `reddit.com/r/{sub}/hot.json` — 10 subreddits, every 12s
* **HackerNews** — Algolia search API — 9 topic queries, every 18s
* **RSS Feeds** — BBC, NPR, Al Jazeera, TechCrunch, The Verge, Ars Technica, Science Daily — every 25s

Posts are deduplicated with MD5 hashing and pushed through an `asyncio.Queue` bridge into the async FastAPI event loop.

### Sentiment Engine

A **weighted ensemble** of three models:

|Model|Weight|Strength|
|-|-|-|
|VADER|30%|Emojis, slang, ALL-CAPS, social media text|
|TextBlob|20%|Fast pattern-based baseline|
|DistilBERT|50%|Context, sarcasm, transformer accuracy|

The final label is the `argmax` of weighted confidence scores across all three models.

### API Endpoints

|Method|Path|Description|
|-|-|-|
|GET|`/`|Health check|
|GET|`/docs`|Swagger UI|
|POST|`/analyze`|Analyse a single text|
|POST|`/analyze/batch`|Analyse up to 50 texts|
|GET|`/history`|Recent DB records|
|GET|`/stats`|Aggregate sentiment counts|
|GET|`/stream/recent`|Last N buffered posts|
|WS|`/ws`|Real-time WebSocket stream|

\---

## 🌐 Deploy Frontend to Netlify

1. Drag `frontend/dashboard.html` to [netlify.com/drop](https://app.netlify.com/drop)
2. Change `API` and `WS\_URL` in the script to point to your deployed backend

\---

## 🔧 Configuration

Edit `.env`:

```
DATABASE\_URL=sqlite:///./sentiment\_data.db
APP\_HOST=0.0.0.0
APP\_PORT=8000
DEBUG=True
```

\---

## Dependencies

* **transformers** — HuggingFace transformer models
* **torch** — PyTorch backend
* **vaderSentiment** — VADER rule-based analyser
* **textblob** — Pattern-based NLP
* **fastapi + uvicorn** — Async web framework
* **sqlalchemy** — ORM for SQLite
* **httpx** — Async HTTP client for data collection
* **loguru** — Structured logging

\---

## &#x20;Planned Enhancements

* Multilingual support via mBERT (104 languages)
* Emotion detection beyond pos/neg (GoEmotions dataset)
* Docker deployment (one-command containerization)
* Topic modelling with LDA
* Email/Slack alerting when sentiment drops below threshold
* Named Entity Recognition with spaCy

