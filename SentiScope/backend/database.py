"""
backend/database.py
SQLAlchemy ORM models, SQLite engine, session management, and aggregate stats queries.
"""

import json
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentiment_data.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── ORM Models ────────────────────────────────────────────────────────────────
class SentimentRecord(Base):
    __tablename__ = "sentiment_records"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String(100), unique=True, index=True)
    source = Column(String(50))
    text = Column(Text)
    author = Column(String(100))
    topic = Column(String(100))
    sentiment = Column(String(20))  # positive / negative / neutral
    confidence = Column(Float)
    pos_score = Column(Float)
    neg_score = Column(Float)
    neu_score = Column(Float)
    likes = Column(Integer, default=0)
    retweets = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    raw_result = Column(Text)  # full JSON result


# ── Database Helper Functions ─────────────────────────────────────────────────
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised")


def get_db() -> Session:
    """Dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_record(db: Session, post: dict, analysis: dict) -> SentimentRecord:
    """Persist a post + its sentiment analysis to the DB."""
    label_scores = analysis.get("label_scores", {})

    # Check for duplicate post_id
    post_id = post.get("id", f"unknown_{int(datetime.now().timestamp())}")
    existing = db.query(SentimentRecord).filter(SentimentRecord.post_id == post_id).first()
    if existing:
        return existing

    record = SentimentRecord(
        post_id=post_id,
        source=post.get("source", "unknown"),
        text=post.get("text", "")[:500],
        author=post.get("author", "anonymous"),
        topic=post.get("topic", "general"),
        sentiment=analysis.get("sentiment", "neutral"),
        confidence=analysis.get("confidence", 0.0),
        pos_score=label_scores.get("positive", 0.0),
        neg_score=label_scores.get("negative", 0.0),
        neu_score=label_scores.get("neutral", 0.0),
        likes=post.get("likes", 0),
        retweets=post.get("retweets", 0),
        raw_result=json.dumps(analysis),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_records(db: Session, limit: int = 100):
    return (
        db.query(SentimentRecord)
        .order_by(SentimentRecord.created_at.desc())
        .limit(limit)
        .all()
    )


def get_sentiment_stats(db: Session) -> dict:
    """Aggregate counts for dashboard summary cards."""
    total = db.query(SentimentRecord).count()
    if total == 0:
        return {"total": 0, "positive": 0, "negative": 0, "neutral": 0, "positivity_rate": 0}

    pos = db.query(SentimentRecord).filter(SentimentRecord.sentiment == "positive").count()
    neg = db.query(SentimentRecord).filter(SentimentRecord.sentiment == "negative").count()
    neu = db.query(SentimentRecord).filter(SentimentRecord.sentiment == "neutral").count()

    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positivity_rate": round(pos / total * 100, 1),
    }
