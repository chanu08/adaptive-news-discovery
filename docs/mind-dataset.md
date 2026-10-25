# MIND Dataset Structure Documentation

> **Source:** Microsoft MIND (MIcrosoft News Dataset)
> **Paper:** Wu et al., ACL 2020 — "MIND: A Large-scale Dataset for News Recommendation"
> **License:** Microsoft Research License Terms (non-commercial research)
> **Download:** https://msnews.github.io

---

## Overview

MIND was collected from **anonymized behavior logs** of the Microsoft News (MSN) website
from **October 12 to November 22, 2019** (6 weeks).

| Version | Users | News Articles | Impression Logs |
|---------|-------|---------------|-----------------|
| MINDlarge (full) | ~1 million | ~160,000 | ~15 million |
| MINDsmall | ~50,000 | subset | subset |

Only training and validation splits contain behaviors.tsv.
The test set contains news.tsv only.

---

## Dataset Splits

| Split | Files | Time Period |
|-------|-------|-------------|
| train | news.tsv, behaviors.tsv, entity_embedding.vec, relation_embedding.vec | First 5 weeks |
| dev (validation) | news.tsv, behaviors.tsv, entity_embedding.vec, relation_embedding.vec | 5th week (last day) |
| test | news.tsv only | 6th week |

---

## File: `news.tsv`

**Format:** Tab-separated values. **NO header row.**
**Encoding:** UTF-8

### Columns (0-indexed)

| Index | Field | Type | Example | Notes |
|-------|-------|------|---------|-------|
| 0 | **NewsID** | string | `N55689` | Unique article ID, pattern `N\d+` |
| 1 | **Category** | string | `lifestyle` | Top-level news category |
| 2 | **SubCategory** | string | `lifestyleroyals` | Fine-grained category |
| 3 | **Title** | string | `The Brands Queen Elizabeth II, Princess Diana...` | Article headline, plain text |
| 4 | **Abstract** | string | `Shop the iconic outfits of the royals.` | Short summary. May be empty string |
| 5 | **URL** | string | `https://msn.com/en-us/lifestyle/...` | Full MSN article URL |
| 6 | **TitleEntities** | JSON string | `[{"Label":"Queen Elizabeth II","Type":"Person",...}]` | Named entities in title. May be `[]` |
| 7 | **AbstractEntities** | JSON string | `[{"Label":"...","Type":"...","WikidataId":"...","Confidence":0.9,...}]` | Named entities in abstract. May be `[]` |

### Entity Object Fields

Each entity in TitleEntities/AbstractEntities is a JSON object with:

| Field | Type | Description |
|-------|------|-------------|
| `Label` | string | Entity display name |
| `Type` | string | Entity type (Person, Location, Organization, etc.) |
| `WikidataId` | string | Wikidata KB identifier (e.g. `Q9682`) |
| `Confidence` | float | Extraction confidence score (0–1) |
| `OccurrenceOffsets` | list[int] | Character offsets in text |
| `SurfaceForms` | list[string] | Surface form variants |

### Known Categories (MINDlarge)

- lifestyle
- health
- news
- sports
- weather
- entertainment
- autos
- travel
- foodanddrink
- finance
- movies
- video
- tv
- music
- kids
- middleeast

### Edge Cases

- **Empty abstract:** ~20% of articles have no abstract (empty string in column 4)
- **Empty entities:** Most articles have `[]` for entity columns
- **URL length:** Some URLs exceed 1000 characters
- **No header row** — do not skip the first line

---

## File: `behaviors.tsv`

**Format:** Tab-separated values. **NO header row.**
**Encoding:** UTF-8

### Columns (0-indexed)

| Index | Field | Type | Example | Notes |
|-------|-------|------|---------|-------|
| 0 | **ImpressionID** | integer | `123456` | Unique per log row (globally unique) |
| 1 | **UserID** | string | `U13740` | Anonymized user ID, pattern `U\d+` |
| 2 | **Time** | string | `11/11/2019 9:43:18 AM` | Format: `M/D/YYYY H:MM:SS AM/PM` |
| 3 | **History** | string | `N3 N12 N42` | Space-separated NewsIDs the user clicked BEFORE this impression. May be empty |
| 4 | **Impressions** | string | `N55689-1 N23554-0 N9134-0` | Space-separated `NewsID-Label` pairs. Label: 1=clicked, 0=not clicked |

### Timestamp Format

```
M/D/YYYY H:MM:SS AM/PM
Examples:
  11/11/2019 9:43:18 AM
  11/12/2019 10:00:00 AM
  11/13/2019 1:00:00 PM
```

Note: Hour is NOT zero-padded (9, not 09).

### Impressions Column

Each token is `NewsID-Label`:
- `N55689-1` → Article N55689 was **clicked** in this impression session
- `N23554-0` → Article N23554 was **shown but not clicked**

The test set omits the `-Label` suffix — labels are to be predicted.

### History Column

- Space-separated NewsIDs of articles the user clicked in the 4-week period BEFORE this impression
- Empty string (`""`) = new user with no prior click history
- Used as the input for user interest modeling in Phase 3+

### Edge Cases

- **Empty history:** Common for new users (~15% of users)
- **Many impressions per log:** Some sessions show 50+ articles
- **Long history:** Heavy users may have 100+ articles in history
- **Test set:** Impressions column has no `-Label` suffix

---

## File: `entity_embedding.vec`

**Format:** Space-separated values (word2vec format)
**First line:** `<num_entities> <embedding_dim>`
**Subsequent lines:** `<WikidataId> <float> <float> ...`

Used for knowledge-graph aware models (not used in Phase 2 base pipeline).
Entities match the `WikidataId` fields in TitleEntities/AbstractEntities.

---

## File: `relation_embedding.vec`

**Format:** Same as entity_embedding.vec
**Content:** Embeddings of relations between entities in the knowledge graph

---

## ID Relationships

```
news.tsv
  └─ NewsID (N\d+)  ─────────────────────────────┐
                                                  │
behaviors.tsv                                     │
  └─ UserID (U\d+)                                │
  └─ History: [NewsID, NewsID, ...]  ─────────────┤
  └─ Impressions: [NewsID-Label, ...]  ────────────┘

entity_embedding.vec
  └─ WikidataId  ←── matches TitleEntities[].WikidataId
                                 and AbstractEntities[].WikidataId
```

---

## Local Directory Layout

```
data/
└── raw/
    └── mind/
        ├── large/
        │   ├── train/        ← extract MINDlarge_train.zip here
        │   │   ├── news.tsv
        │   │   ├── behaviors.tsv
        │   │   ├── entity_embedding.vec
        │   │   └── relation_embedding.vec
        │   ├── dev/          ← extract MINDlarge_dev.zip here
        │   │   ├── news.tsv
        │   │   ├── behaviors.tsv
        │   │   ├── entity_embedding.vec
        │   │   └── relation_embedding.vec
        │   └── test/         ← extract MINDlarge_test.zip here (optional)
        │       └── news.tsv
        └── small/            ← optional MINDsmall for quick testing
            ├── train/
            └── dev/
```

---

## Database Schema (Phase 2)

| Table | Source | Key Fields |
|-------|--------|-----------|
| `news_articles` | news.tsv | `mind_id` (UNIQUE), `category`, `split` |
| `mind_users` | behaviors.tsv | `mind_user_id` (UNIQUE) |
| `behavior_logs` | behaviors.tsv | `impression_id`, `user_mind_id`, `time`, `split` |
| `impression_items` | behaviors.tsv Impressions | `behavior_log_id` (FK), `article_mind_id`, `clicked` |
| `ingestion_runs` | pipeline audit | `run_id`, `split`, `status`, counts |

---

## Ingestion Command

```bash
# From backend/ directory
python -m app.data.mind_ingestion ../data/raw/mind/large/train train
python -m app.data.mind_ingestion ../data/raw/mind/large/dev dev
```

Or use the frontend Dataset page → Trigger ingestion panel.

---

## References

- Paper: https://msnews.github.io/assets/doc/ACL2020_MIND.pdf
- Official site: https://msnews.github.io
- Format doc: https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
