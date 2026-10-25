# Phase 17: Repository Readiness Recommendation

The Adaptive News Discovery project has grown into a robust, full-scale research codebase. To prepare it for publication or deposit into a research repository (such as Zenodo, GitHub, or institutional archives), we recommend formalizing the directory structure. 

**Recommended Repository Structure:**
```text
adaptive-news-discovery/
├── backend/
│   ├── app/                # FastAPI application core
│   ├── data/               # Empty directory for DB and matrices (git-ignored)
│   ├── tests/              # Pytest suite
│   ├── main.py             # Entrypoint
│   └── requirements.txt    # Frozen Python dependencies
├── frontend/
│   ├── src/                # React/Vanilla UI
│   ├── public/             # Static assets
│   └── package.json        # NPM dependencies
├── scripts/
│   ├── ingestion/          # MIND parsing scripts
│   └── evaluation/         # Phase 11-16 evaluation orchestrators
├── reports/
│   ├── phase11_*.md        # Research reports
│   └── phase17_*.md
├── docs/
│   ├── architecture.md     # System design
│   └── experiment_manifest.md
├── .gitignore              # Ignores large data files and matrices
└── README.md               # Unified setup guide
```

### Justification
This structure strictly separates the *application implementation* (backend/frontend) from the *research orchestration* (scripts/reports). It prevents raw dataset files from being accidentally committed by using a dedicated, ignored `data/` directory.

### Data Handling Rule
**IMPORTANT:** The Microsoft MIND dataset is subject to specific licensing terms. The `data/` directory MUST be `.gitignore`'d. Raw TSV files, the SQLite database `news_discovery.db`, and pickled TF-IDF/NMF matrices should NEVER be pushed to a public source repository. Researchers must download the MIND dataset independently from Microsoft and run the ingestion scripts locally.
