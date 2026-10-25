# Backend — Adaptive News Discovery API

FastAPI backend for the Adaptive News Discovery IR project.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs  
Health:   http://localhost:8000/api/health

## Project Layout

```
app/
├── main.py          # FastAPI entry point, CORS, routers
├── config/          # Pydantic-settings env config
├── database/        # Async SQLAlchemy engine + session
├── models/          # ORM table models
├── schemas/         # Pydantic request/response schemas
├── routes/          # FastAPI routers
└── services/        # IR algorithm modules (placeholders Phase 1)
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```
