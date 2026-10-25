"""
MIND Dataset Parser — Phase 2

Low-level parser for raw MIND .tsv files.
Produces plain Python dataclasses — no database dependency.

MIND news.tsv columns (0-indexed, tab-separated, NO header row):
  0  NewsID          e.g. "N55689"
  1  Category        e.g. "lifestyle"
  2  SubCategory     e.g. "lifestyleroyals"
  3  Title           plain text
  4  Abstract        plain text (may be empty string)
  5  URL             full MSN article URL
  6  TitleEntities   JSON array of entity dicts (may be "[]")
  7  AbstractEntities JSON array of entity dicts (may be "[]")

MIND behaviors.tsv columns (0-indexed, tab-separated, NO header row):
  0  ImpressionID    integer
  1  UserID          e.g. "U13740"
  2  Time            e.g. "11/11/2019 9:43:18 AM"
  3  History         space-separated NewsIDs (may be empty string)
  4  Impressions     space-separated "NewsID-Label" pairs (Label: 1=click, 0=no-click)

Both files have NO header row.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ── MIND timestamp format ─────────────────────────────────────────────────────
_MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses (pure Python, no SQLAlchemy dependency)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawNewsRecord:
    mind_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str    # raw JSON string
    abstract_entities: str # raw JSON string
    line_number: int = 0


@dataclass
class ParsedNewsRecord:
    mind_id: str
    category: str
    subcategory: str
    title: str
    abstract: Optional[str]
    url: Optional[str]
    title_entities: str    # normalized JSON string
    abstract_entities: str # normalized JSON string


@dataclass
class ImpressionEntry:
    article_mind_id: str
    clicked: int   # 1 or 0


@dataclass
class RawBehaviorRecord:
    impression_id: int
    user_mind_id: str
    time_raw: str
    history_raw: str
    impressions_raw: str
    line_number: int = 0


@dataclass
class ParsedBehaviorRecord:
    impression_id: int
    user_mind_id: str
    time: Optional[datetime]
    history: list[str]          # list of NewsIDs
    impressions: list[ImpressionEntry]
    history_length: int = 0
    n_impressions: int = 0
    n_clicks: int = 0

    def __post_init__(self) -> None:
        self.history_length = len(self.history)
        self.n_impressions  = len(self.impressions)
        self.n_clicks       = sum(i.clicked for i in self.impressions)


@dataclass
class ParseResult:
    """Summary of a parse pass over one file."""
    file_path: str
    total_lines: int = 0
    parsed_ok: int   = 0
    skipped: int     = 0
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timestamp(raw: str) -> Optional[datetime]:
    """Parse MIND timestamp string to datetime.  Returns None on failure."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    try:
        return datetime.strptime(raw, _MIND_TIME_FORMAT)
    except ValueError:
        # Try with single-digit hour (some rows miss the leading zero)
        try:
            return datetime.strptime(raw, "%-m/%d/%Y %I:%M:%S %p")
        except ValueError:
            pass
    logger.debug("Cannot parse timestamp: %r", raw)
    return None


def _normalize_json(raw: str) -> str:
    """Validate and re-serialize JSON field.  Returns '[]' on invalid input."""
    if not raw or not raw.strip():
        return "[]"
    try:
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return "[]"


def _parse_impressions(raw: str) -> list[ImpressionEntry]:
    """
    Parse the Impressions column.
    Each token is "NewsID-Label" e.g. "N1234-1" or "N5678-0".
    Test-set rows may omit labels → treated as label=-1 (unknown).
    """
    if not raw or not raw.strip():
        return []
    entries: list[ImpressionEntry] = []
    for token in raw.strip().split():
        parts = token.rsplit("-", 1)
        if len(parts) == 2:
            news_id, label_str = parts
            try:
                label = int(label_str)
            except ValueError:
                label = -1
        else:
            news_id = parts[0]
            label = -1
        if news_id:
            entries.append(ImpressionEntry(article_mind_id=news_id, clicked=label))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# News Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_news_file(
    path: Path,
    *,
    skip_invalid: bool = True,
) -> Iterator[ParsedNewsRecord | Exception]:
    """
    Stream-parse a MIND news.tsv file line by line.

    Yields either ParsedNewsRecord on success, or an Exception (with a
    `line_number` attribute) on failure.  Caller decides whether to skip or abort.

    Parameters
    ----------
    path        : Path to news.tsv
    skip_invalid: If True, log and yield Exception; if False, re-raise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIND news file not found: {path}")

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for lineno, row in enumerate(reader, start=1):
            try:
                if len(row) < 5:
                    raise ValueError(
                        f"Line {lineno}: expected ≥5 columns, got {len(row)}: {row!r}"
                    )

                mind_id     = row[0].strip()
                category    = row[1].strip()
                subcategory = row[2].strip() if len(row) > 2 else ""
                title       = row[3].strip() if len(row) > 3 else ""
                abstract    = row[4].strip() if len(row) > 4 else ""
                url         = row[5].strip() if len(row) > 5 else ""
                title_ents  = _normalize_json(row[6]) if len(row) > 6 else "[]"
                abs_ents    = _normalize_json(row[7]) if len(row) > 7 else "[]"

                if not mind_id:
                    raise ValueError(f"Line {lineno}: empty NewsID")
                if not title:
                    raise ValueError(f"Line {lineno}: empty title (mind_id={mind_id!r})")

                yield ParsedNewsRecord(
                    mind_id=mind_id,
                    category=category or "unknown",
                    subcategory=subcategory,
                    title=title,
                    abstract=abstract or None,
                    url=url or None,
                    title_entities=title_ents,
                    abstract_entities=abs_ents,
                )

            except Exception as exc:
                exc.line_number = lineno  # type: ignore[attr-defined]
                if skip_invalid:
                    yield exc
                else:
                    raise


# ─────────────────────────────────────────────────────────────────────────────
# Behavior Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_behaviors_file(
    path: Path,
    *,
    skip_invalid: bool = True,
) -> Iterator[ParsedBehaviorRecord | Exception]:
    """
    Stream-parse a MIND behaviors.tsv file line by line.

    Yields either ParsedBehaviorRecord on success, or an Exception on failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIND behaviors file not found: {path}")

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for lineno, row in enumerate(reader, start=1):
            try:
                if len(row) < 2:
                    raise ValueError(
                        f"Line {lineno}: expected ≥2 columns, got {len(row)}: {row!r}"
                    )

                impression_id_raw = row[0].strip()
                user_mind_id      = row[1].strip()
                time_raw          = row[2].strip() if len(row) > 2 else ""
                history_raw       = row[3].strip() if len(row) > 3 else ""
                impressions_raw   = row[4].strip() if len(row) > 4 else ""

                if not impression_id_raw:
                    raise ValueError(f"Line {lineno}: empty ImpressionID")
                if not user_mind_id:
                    raise ValueError(f"Line {lineno}: empty UserID")

                try:
                    impression_id = int(impression_id_raw)
                except ValueError:
                    raise ValueError(
                        f"Line {lineno}: ImpressionID not an integer: {impression_id_raw!r}"
                    )

                history = (
                    [nid for nid in history_raw.split() if nid]
                    if history_raw
                    else []
                )
                impressions = _parse_impressions(impressions_raw)
                ts = _parse_timestamp(time_raw)

                yield ParsedBehaviorRecord(
                    impression_id=impression_id,
                    user_mind_id=user_mind_id,
                    time=ts,
                    history=history,
                    impressions=impressions,
                )

            except Exception as exc:
                exc.line_number = lineno  # type: ignore[attr-defined]
                if skip_invalid:
                    yield exc
                else:
                    raise
