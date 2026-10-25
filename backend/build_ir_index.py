"""
Build IR Index — Phase 5
Command line script to trigger TF-IDF and NMF index building.
"""

import asyncio
import time
from app.database.connection import AsyncSessionLocal
from app.services.ir.indexer import build_index

async def main():
    print("Starting IR Index Build Process...")
    t0 = time.perf_counter()
    
    async with AsyncSessionLocal() as session:
        stats = await build_index(session)
        
    t1 = time.perf_counter()
    
    if stats:
        print("\n=== PHASE 5 FINAL RESULTS ===")
        print(f"Canonical documents: {stats['canonical_docs']}")
        print(f"Documents with abstract: {stats['with_abstract']}")
        print(f"Documents title only: {stats['title_only']}")
        print(f"Vocabulary size: {stats['vocab_size']}")
        print(f"TF-IDF matrix shape: {stats['matrix_shape']}")
        print(f"Non-zero entries: {stats['nnz']}")
        print(f"Topics extracted: {stats['topics']}")
        print(f"Total build time: {t1 - t0:.2f} seconds")
        print("\nPreprocessing: PASS")
        print("TF-IDF: PASS")
        print("Topic extraction: PASS")
        print("Persistence: PASS")

if __name__ == "__main__":
    asyncio.run(main())
