"""
Adaptive News Discovery - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.database.connection import init_db
from app.routes import health, articles, users, search, dataset, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    print("Database initialized successfully.")
    yield
    # Shutdown
    print("Shutting down application.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Adaptive News Discovery Using Interest Drift and "
        "Diversity-Aware Information Retrieval — Backend API"
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow local frontend dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router,    prefix="/api",          tags=["Health"])
app.include_router(articles.router,  prefix="/api/articles", tags=["Articles"])
app.include_router(users.router,     prefix="/api/users",    tags=["Users"])
app.include_router(feedback.router,  prefix="/api/users",    tags=["Feedback"])
app.include_router(search.router,    prefix="/api/search",   tags=["Search"])
app.include_router(dataset.router,   prefix="/api/dataset",  tags=["Dataset"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
