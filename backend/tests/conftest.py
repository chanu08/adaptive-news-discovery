"""
Test configuration — conftest.py

Initialises the in-memory async SQLite database before any test that
requires it (articles endpoint, stats endpoint, etc.)
"""
import asyncio
import pytest


# ── pytest-asyncio configuration ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


# ── Database initialisation ──────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def initialize_test_db():
    """
    Create all ORM tables in the test database before any test runs.
    This is synchronous to keep compatibility across pytest-asyncio versions.
    """
    async def _init():
        from app.database.connection import init_db
        await init_db()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_init())
    finally:
        loop.close()
