"""
Application settings loaded from environment variables / .env file.
Uses pydantic-settings for validation.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────
    APP_NAME: str = "Adaptive News Discovery"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── Database ────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/news_discovery.db"

    # ── CORS ────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA / misc
        "http://127.0.0.1:5173",
    ]

    # ── News API (placeholder — populated in a later phase) ─────────────────
    NEWS_API_KEY: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
