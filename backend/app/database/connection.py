"""
Async SQLite connection layer using SQLAlchemy + aiosqlite.

Only the engine and session factory are set up here.
Individual table models are defined in app/models/ and imported
during init_db() so each phase can add new tables independently.
"""

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import settings

# ---------------------------------------------------------------------------
# Base class shared by all ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Engine — created once at module import time
# ---------------------------------------------------------------------------

# Ensure the data directory exists before SQLite tries to create the file
_db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
os.makedirs(os.path.dirname(_db_path) if os.path.dirname(_db_path) else ".", exist_ok=True)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Force False to prevent massive slowdown during ingestion
    connect_args={"check_same_thread": False},
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Dependency — injected into FastAPI route handlers
# ---------------------------------------------------------------------------

async def get_db() -> AsyncSession:
    """Yield a database session and close it when the request ends."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create all tables whose models have been imported.

    Import every model module here so SQLAlchemy knows about them
    before calling create_all().
    """
    # Phase 1 + Phase 2 — import all model modules so SQLAlchemy knows about them
    from app.models import article, user  # noqa: F401  Phase 1
    from app.models import (              # noqa: F401  Phase 2
        news_article,
        mind_user,
        behavior,
        ingestion_run,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
