"""Read environment settings and define defaults for the travel assistant."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Use .env NEXT TO this file, even if the app was started from another folder.
# Variables already set in the terminal/hosting environment take precedence.
load_dotenv(Path(__file__).with_name(".env"), override=False)
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()

AUTH_TIMEOUT = 15
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().with_name("knowledge_base.json")
EMBEDDING_MODEL = "embeddinggemma"
CHAT_MODEL = "qwen3:4b"
TOP_K = 3
MIN_SIMILARITY = 0.30
OLLAMA_TIMEOUT = 300
