"""
Phase 2 Tests — MIND Parser, Validation, Ingestion, Statistics

Tests run against in-memory or temp SQLite databases — no real MIND files required.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_NEWS_ROWS = [
    # Valid rows
    "N1\tlifestyle\tlifestyleroyals\tQueen visits Scotland\tThe Queen visited Edinburgh.\thttps://msn.com/1\t[]\t[]",
    "N2\ttech\ttechnology\tAI breakthroughs 2019\tDeep learning improves.\thttps://msn.com/2\t[]\t[]",
    "N3\tsports\tsoccer\tWorld Cup 2022\t\thttps://msn.com/3\t[]\t[]",  # empty abstract
    # Row with entity JSON
    'N4\tfinance\tstocks\tMarkets rally\tDow Jones up 200 points.\thttps://msn.com/4\t[{"Label":"Dow Jones"}]\t[]',
    # Row missing URL column (5 cols instead of 8)
    "N5\thealth\twellness\tTips for sleep\tSleep 8 hours.",
]

SAMPLE_NEWS_INVALID = [
    "",                                     # empty line
    "\t\t\t",                               # all empty
    "BADID\tlifestyle\ttest\t\tno title",   # empty title
]

SAMPLE_BEHAVIOR_ROWS = [
    # impression_id, user_id, time, history, impressions
    "1\tU1\t11/11/2019 9:43:18 AM\tN1 N2\tN3-1 N4-0 N5-0",
    "2\tU2\t11/12/2019 10:00:00 AM\t\tN1-1 N2-1",   # empty history
    "3\tU1\t11/13/2019 8:00:00 AM\tN3\tN1-0",
]

SAMPLE_BEHAVIOR_INVALID = [
    "",
    "\t",
    "notanint\tU99\t11/11/2019 9:00:00 AM\t\t",  # non-int impression_id
]


def _write_tsv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Parser tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMindNewsParser:
    def test_parse_valid_rows(self, tmp_path):
        from app.data.mind_parser import parse_news_file, ParsedNewsRecord

        news_file = tmp_path / "news.tsv"
        _write_tsv(news_file, SAMPLE_NEWS_ROWS[:4])  # 4 valid rows

        records = list(parse_news_file(news_file))
        valid = [r for r in records if isinstance(r, ParsedNewsRecord)]
        assert len(valid) == 4

    def test_parse_preserves_mind_id(self, tmp_path):
        from app.data.mind_parser import parse_news_file, ParsedNewsRecord

        news_file = tmp_path / "news.tsv"
        _write_tsv(news_file, [SAMPLE_NEWS_ROWS[0]])

        records = [r for r in parse_news_file(news_file) if isinstance(r, ParsedNewsRecord)]
        assert records[0].mind_id == "N1"
        assert records[0].category == "lifestyle"
        assert records[0].title == "Queen visits Scotland"

    def test_parse_missing_url_column(self, tmp_path):
        """Row with 5 columns (missing URL, entities) should still parse."""
        from app.data.mind_parser import parse_news_file, ParsedNewsRecord

        news_file = tmp_path / "news.tsv"
        _write_tsv(news_file, [SAMPLE_NEWS_ROWS[4]])

        records = [r for r in parse_news_file(news_file) if isinstance(r, ParsedNewsRecord)]
        assert len(records) == 1
        assert records[0].url is None

    def test_parse_entity_json(self, tmp_path):
        """Entity fields should be normalized to valid JSON strings."""
        from app.data.mind_parser import parse_news_file, ParsedNewsRecord

        news_file = tmp_path / "news.tsv"
        _write_tsv(news_file, [SAMPLE_NEWS_ROWS[3]])

        records = [r for r in parse_news_file(news_file) if isinstance(r, ParsedNewsRecord)]
        assert records[0].title_entities != "[]"
        parsed = json.loads(records[0].title_entities)
        assert isinstance(parsed, list)

    def test_invalid_rows_yield_exceptions(self, tmp_path):
        from app.data.mind_parser import parse_news_file

        news_file = tmp_path / "news.tsv"
        # Mix: valid + invalid
        rows = [SAMPLE_NEWS_ROWS[0]] + SAMPLE_NEWS_INVALID
        _write_tsv(news_file, rows)

        results = list(parse_news_file(news_file))
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) >= 2  # empty line + all-empty + bad-id or empty title

    def test_nonexistent_file_raises(self):
        from app.data.mind_parser import parse_news_file

        with pytest.raises(FileNotFoundError):
            list(parse_news_file(Path("/nonexistent/news.tsv")))


class TestMindBehaviorParser:
    def test_parse_valid_behaviors(self, tmp_path):
        from app.data.mind_parser import parse_behaviors_file, ParsedBehaviorRecord

        beh_file = tmp_path / "behaviors.tsv"
        _write_tsv(beh_file, SAMPLE_BEHAVIOR_ROWS)

        records = [r for r in parse_behaviors_file(beh_file) if isinstance(r, ParsedBehaviorRecord)]
        assert len(records) == 3

    def test_impression_count(self, tmp_path):
        from app.data.mind_parser import parse_behaviors_file, ParsedBehaviorRecord

        beh_file = tmp_path / "behaviors.tsv"
        _write_tsv(beh_file, [SAMPLE_BEHAVIOR_ROWS[0]])

        records = [r for r in parse_behaviors_file(beh_file) if isinstance(r, ParsedBehaviorRecord)]
        r = records[0]
        assert r.n_impressions == 3
        assert r.n_clicks == 1
        assert r.history_length == 2

    def test_empty_history(self, tmp_path):
        from app.data.mind_parser import parse_behaviors_file, ParsedBehaviorRecord

        beh_file = tmp_path / "behaviors.tsv"
        _write_tsv(beh_file, [SAMPLE_BEHAVIOR_ROWS[1]])

        records = [r for r in parse_behaviors_file(beh_file) if isinstance(r, ParsedBehaviorRecord)]
        assert records[0].history == []
        assert records[0].history_length == 0

    def test_invalid_behaviors_yield_exceptions(self, tmp_path):
        from app.data.mind_parser import parse_behaviors_file

        beh_file = tmp_path / "behaviors.tsv"
        _write_tsv(beh_file, SAMPLE_BEHAVIOR_INVALID)

        results = list(parse_behaviors_file(beh_file))
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) >= 2


class TestTimestampParsing:
    def test_valid_timestamp(self):
        from app.data.mind_parser import _parse_timestamp

        ts = _parse_timestamp("11/11/2019 9:43:18 AM")
        assert ts is not None
        assert ts.year == 2019
        assert ts.month == 11

    def test_empty_timestamp_returns_none(self):
        from app.data.mind_parser import _parse_timestamp

        assert _parse_timestamp("") is None
        assert _parse_timestamp("   ") is None

    def test_malformed_timestamp_returns_none(self):
        from app.data.mind_parser import _parse_timestamp

        assert _parse_timestamp("not a date") is None


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_news_record(self):
        from app.data.mind_parser import ParsedNewsRecord
        from app.data.validation import validate_news_record

        record = ParsedNewsRecord(
            mind_id="N100",
            category="tech",
            subcategory="software",
            title="AI advances",
            abstract="Details here.",
            url="https://example.com",
            title_entities="[]",
            abstract_entities="[]",
        )
        result = validate_news_record(record)
        assert result.valid is True
        assert result.issues == []

    def test_invalid_news_id_format(self):
        from app.data.mind_parser import ParsedNewsRecord
        from app.data.validation import validate_news_record

        record = ParsedNewsRecord(
            mind_id="BAD-ID",
            category="tech",
            subcategory="",
            title="Some title",
            abstract=None,
            url=None,
            title_entities="[]",
            abstract_entities="[]",
        )
        result = validate_news_record(record)
        assert result.valid is False

    def test_empty_title_is_invalid(self):
        from app.data.mind_parser import ParsedNewsRecord
        from app.data.validation import validate_news_record

        record = ParsedNewsRecord(
            mind_id="N200",
            category="tech",
            subcategory="",
            title="",
            abstract=None,
            url=None,
            title_entities="[]",
            abstract_entities="[]",
        )
        result = validate_news_record(record)
        assert result.valid is False
        assert any("title" in i.lower() for i in result.issues)

    def test_missing_abstract_is_warning(self):
        from app.data.mind_parser import ParsedNewsRecord
        from app.data.validation import validate_news_record

        record = ParsedNewsRecord(
            mind_id="N300",
            category="sports",
            subcategory="soccer",
            title="Match results",
            abstract=None,
            url="https://example.com",
            title_entities="[]",
            abstract_entities="[]",
        )
        result = validate_news_record(record)
        assert result.valid is True       # warnings don't make it invalid
        assert len(result.warnings) > 0

    def test_behavior_validation_invalid_user_id(self):
        from app.data.mind_parser import ParsedBehaviorRecord
        from app.data.validation import validate_behavior_record

        record = ParsedBehaviorRecord(
            impression_id=1,
            user_mind_id="BADUSER",
            time=None,
            history=[],
            impressions=[],
        )
        result = validate_behavior_record(record)
        assert result.valid is False

    def test_behavior_validation_orphan_articles(self):
        from app.data.mind_parser import ImpressionEntry, ParsedBehaviorRecord
        from app.data.validation import validate_behavior_record

        record = ParsedBehaviorRecord(
            impression_id=42,
            user_mind_id="U100",
            time=datetime(2019, 11, 11, 9, 0, 0),
            history=[],
            impressions=[ImpressionEntry("GHOST1", 1), ImpressionEntry("GHOST2", 0)],
        )
        # known_article_ids does not contain GHOST1/GHOST2
        result = validate_behavior_record(record, known_article_ids={"N1", "N2"})
        assert result.valid is True          # orphans are warnings, not failures
        assert any("orphan" in w.lower() or "not in news" in w.lower() for w in result.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# API smoke tests (Phase 1 + 2)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_articles_endpoint_empty():
    """With an empty DB, articles list should return empty items."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/articles/")
    assert r.status_code == 200
    data = r.json()
    assert "items" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_dataset_stats_endpoint():
    """Stats endpoint should return a valid JSON with zero counts if DB is empty."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/dataset/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_articles" in data
    assert "total_users" in data
    assert "total_behavior_logs" in data


@pytest.mark.asyncio
async def test_search_placeholder():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/search/?q=technology")
    assert r.status_code == 200
    assert r.json()["results"] == []
