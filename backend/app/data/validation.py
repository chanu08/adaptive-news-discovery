"""
MIND Data Validation — Phase 2

Validation checks applied BEFORE database insertion.
Each check returns a ValidationResult; failures are logged but non-fatal
unless marked as critical.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.data.mind_parser import ParsedNewsRecord, ParsedBehaviorRecord

logger = logging.getLogger(__name__)

# ── Valid MIND ID patterns ────────────────────────────────────────────────────
_NEWS_ID_RE  = re.compile(r"^N\d+$")
_USER_ID_RE  = re.compile(r"^U\d+$")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid: bool
    record_id: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.valid = False
        self.issues.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# News article validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_news_record(record: ParsedNewsRecord) -> ValidationResult:
    """
    Validate a single parsed news record.

    Critical failures (valid=False):
      - Non-matching NewsID pattern
      - Empty title

    Warnings (logged but allowed):
      - Empty abstract
      - Empty URL
      - Empty category
    """
    result = ValidationResult(valid=True, record_id=record.mind_id)

    # ID format
    if not _NEWS_ID_RE.match(record.mind_id):
        result.add_issue(f"NewsID does not match pattern N<digits>: {record.mind_id!r}")

    # Required content
    if not record.title or not record.title.strip():
        result.add_issue("title is empty")

    if len(record.title) > 1000:
        result.add_issue(f"title too long ({len(record.title)} chars, max 1000)")

    # Warnings
    if not record.abstract:
        result.add_warning("abstract is empty")

    if not record.url:
        result.add_warning("url is missing")

    if not record.category or record.category == "unknown":
        result.add_warning("category is missing or unknown")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Behavior record validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_behavior_record(
    record: ParsedBehaviorRecord,
    known_article_ids: Optional[set[str]] = None,
) -> ValidationResult:
    """
    Validate a single parsed behavior record.

    Parameters
    ----------
    record            : The parsed behavior record to validate.
    known_article_ids : Optional set of valid article MIND IDs.
                        When provided, orphan references are flagged as warnings.

    Critical failures:
      - ImpressionID not positive
      - UserID does not match pattern

    Warnings:
      - timestamp is None (missing/unparseable)
      - impressions list is empty
      - article references not in known_article_ids
    """
    result = ValidationResult(
        valid=True, record_id=str(record.impression_id)
    )

    # ImpressionID must be a positive integer
    if record.impression_id <= 0:
        result.add_issue(
            f"ImpressionID must be positive, got {record.impression_id}"
        )

    # UserID format
    if not _USER_ID_RE.match(record.user_mind_id):
        result.add_issue(
            f"UserID does not match pattern U<digits>: {record.user_mind_id!r}"
        )

    # Timestamp
    if record.time is None:
        result.add_warning("timestamp could not be parsed (row will be stored with NULL time)")

    # Empty impressions (test-set rows may be valid)
    if not record.impressions:
        result.add_warning("impressions list is empty")

    # Orphan article references
    if known_article_ids is not None:
        orphans = [
            imp.article_mind_id
            for imp in record.impressions
            if imp.article_mind_id not in known_article_ids
        ]
        if orphans:
            result.add_warning(
                f"{len(orphans)} impression article(s) not in news index: "
                f"{orphans[:5]}{'...' if len(orphans) > 5 else ''}"
            )

        hist_orphans = [
            nid for nid in record.history if nid not in known_article_ids
        ]
        if hist_orphans:
            result.add_warning(
                f"{len(hist_orphans)} history article(s) not in news index"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationSummary:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    warning_only: int = 0
    issue_counts: dict[str, int] = field(default_factory=dict)

    def record(self, result: ValidationResult) -> None:
        self.total += 1
        if result.valid:
            self.valid += 1
            if result.warnings:
                self.warning_only += 1
        else:
            self.invalid += 1
            for issue in result.issues:
                key = issue.split(":")[0][:60]
                self.issue_counts[key] = self.issue_counts.get(key, 0) + 1

    def __str__(self) -> str:
        return (
            f"ValidationSummary(total={self.total}, valid={self.valid}, "
            f"invalid={self.invalid}, warning_only={self.warning_only})"
        )
