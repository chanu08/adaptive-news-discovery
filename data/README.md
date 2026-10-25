# Data Directory

This directory stores all MIND dataset files and processed outputs.

## Directory Structure

data/
├── raw/
│   └── mind/
│       ├── large/
│       │   ├── train/        ← Extract MINDlarge_train.zip here
│       │   ├── dev/          ← Extract MINDlarge_dev.zip here
│       │   └── test/         ← Extract MINDlarge_test.zip here (optional)
│       └── small/            ← Optional MINDsmall for quick testing
│           ├── train/
│           └── dev/
├── processed/
│   └── mind/                 ← Post-pipeline processed data
└── indices/                  ← TF-IDF and search indices (Phase 3+)

## Download Instructions

1. Visit: https://msnews.github.io
2. Accept the Microsoft Research License Terms
3. Download MINDlarge_train.zip and MINDlarge_dev.zip
4. Extract into data/raw/mind/large/train/ and data/raw/mind/large/dev/

DO NOT commit raw data files to git. They are excluded via .gitignore.

## Ingestion

After placing files, trigger ingestion via:
  - Frontend: Dataset page → Ingestion panel
  - CLI: cd backend && python -m app.data.mind_ingestion ../data/raw/mind/large/train train

See docs/mind-dataset.md for complete format documentation.
