"""
backend/main.py
FastAPI application with:
• REST endpoints for analysis and history
• WebSocket endpoint for real-time streaming
• Background task that feeds the sentiment pipeline
"""

import json
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from collections import deque
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_model import EnsembleSentimentAnalyzer
from backend.database import init_db, get_db, save_record, get_recent_records, get_sentiment_stats
from backend.data_collector import create_collector

# ── Global state ──────────────────────────────────────────────────────────────
analyzer = EnsembleSentimentAnalyzer(use_transformer=True)
event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
recent_buffer: deque = deque(maxlen=200)
active_ws: Set[WebSocket] = set()
collector = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global collector
    init_db()

    loop = asyncio.get_running_loop()

    def on_post(post: dict):
        try:
            analysis = analyzer.analyze(post["text"])
            enriched = {**post, **analysis, "timestamp": post.get("timestamp")}
            recent_buffer.appendleft(enriched)
            loop.call_soon_threadsafe(event_queue.put_nowait, enriched)
        except Exception as exc:
            logger.error(f"Analysis error: {exc}")

    collector = create_collector("multi", on_post)
    collector.start()
    logger.info("Application startup complete — streaming Reddit + HN + RSS")

    task = asyncio.create_task(_broadcaster())

    yield  # app is running

    task.cancel()
    if collector:
        collector.stop()
    logger.info("Application shutdown")


async def _broadcaster():
    """Pull items from queue and fan-out to all connected WebSocket clients."""
    while True:
        try:
            item = await asyncio.wait_for(event_queue.get(), timeout=1.0)
            dead: Set[WebSocket] = set()
            for ws in list(active_ws):
                try:
                    await ws.send_text(json.dumps(item, default=str))
                except Exception:
                    dead.add(ws)
            active_ws.difference_update(dead)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Broadcaster error: {exc}")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Sentiment Analysis API",
    description="Live sentiment from Reddit, HackerNews & RSS — no API keys needed",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"

class BatchRequest(BaseModel):
    texts: list


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API v2", "status": "running", "docs": "/docs"}


@app.get("/sources")
def get_sources():
    return {
        "sources": [
            {"name": "Reddit", "api": "reddit.com/r/{sub}/hot.json", "auth": False, "interval_s": 12},
            {"name": "HackerNews", "api": "hn.algolia.com/api/v1/search_by_date", "auth": False, "interval_s": 18},
            {"name": "NewsRSS", "api": "BBC/NPR/AlJazeera/TechCrunch/TheVerge/ArsTechnica RSS", "auth": False, "interval_s": 25},
        ],
        "note": "All sources are public and require no API keys."
    }


@app.post("/analyze")
def analyze_text(req: TextRequest, db: Session = Depends(get_db)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = analyzer.analyze(req.text)
    post = {
        "id": f"manual_{int(datetime.now().timestamp()*1000)}",
        "source": req.source,
        "text": req.text,
        "author": "api_user",
        "topic": "manual",
        "likes": 0,
        "retweets": 0,
    }
    save_record(db, post, result)
    return result


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    if len(req.texts) > 50:
        raise HTTPException(status_code=400, detail="Max 50 texts per batch")
    return analyzer.batch_analyze(req.texts)


@app.get("/history")
def get_history(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    records = get_recent_records(db, limit)
    return [
        {
            "id": r.id,
            "post_id": r.post_id,
            "source": r.source,
            "text": r.text[:140],
            "sentiment": r.sentiment,
            "confidence": r.confidence,
            "topic": r.topic,
            "timestamp": r.created_at.isoformat(),
            "likes": r.likes,
        }
        for r in records
    ]


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    stats = get_sentiment_stats(db)
    stats["recent_count"] = len(recent_buffer)
    stats["active_connections"] = len(active_ws)
    return stats


@app.get("/stream/recent")
def get_recent_stream(limit: int = Query(20, ge=1, le=200)):
    return list(recent_buffer)[:limit]


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_ws.add(ws)
    logger.info(f"WebSocket connected. Active: {len(active_ws)}")

    for item in list(recent_buffer)[:10]:
        try:
            await ws.send_text(json.dumps(item, default=str))
        except Exception:
            break

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        active_ws.discard(ws)
        logger.info(f"WebSocket disconnected. Active: {len(active_ws)}")
