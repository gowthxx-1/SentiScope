#!/bin/bash
# SentimentPulse — Quick Start Script

set -e

echo "🔧 Setting up SentimentPulse..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install deps
echo "📥 Installing dependencies..."
pip install -r requirements.txt -q

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Starting backend on http://localhost:8000"
echo "📊 Open frontend/dashboard.html in your browser"
echo "📚 API docs at http://localhost:8000/docs"
echo ""

uvicorn backend.main:app --reload --port 8000
