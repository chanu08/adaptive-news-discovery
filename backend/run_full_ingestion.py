import asyncio
import logging
from pathlib import Path
from app.database.connection import AsyncSessionLocal, init_db
from sqlalchemy import text
from app.data.mind_ingestion import ingest_split
import time
import sys

async def main():
    # Make sure we init db
    await init_db()

    base = Path("../data/extracted/mind")
    splits = ["dev", "train", "test"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Less spam
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    overall_start = time.time()
    
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            print(f"Directory {split_dir} does not exist. Skipping {split}.")
            continue
            
        print(f"\n{'='*60}\nStarting ingestion for {split}\n{'='*60}")
        async with AsyncSessionLocal() as session:
            await session.execute(text("PRAGMA synchronous = OFF;"))
            await session.execute(text("PRAGMA journal_mode = MEMORY;"))
            try:
                run = await ingest_split(split_dir, split, session)
                await session.commit()
                print(f"\nSuccessfully ingested {split}")
                print(f"  Duration: {run.duration_seconds:.1f}s")
                print(f"  News inserted: {run.news_inserted}")
                print(f"  Behaviors inserted: {run.behaviors_inserted}")
                print(f"  Items inserted: {run.items_inserted}")
                print(f"  Users inserted: {run.users_inserted}")
            except Exception as e:
                print(f"\nFAILED to ingest {split}: {e}")

    overall_duration = time.time() - overall_start
    print(f"\nAll specified splits processed in {overall_duration:.1f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
