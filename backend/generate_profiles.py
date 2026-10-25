"""
Batch Profile Generator — Phase 3
Generates user interest profiles for MIND users in bulk.
"""

import asyncio
import argparse
import logging
import time
from sqlalchemy import select, func
from app.database.connection import AsyncSessionLocal
from app.models.mind_user import MindUser
from app.services.interest_modeling.profile_builder import build_user_profiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def get_user_batch(session, offset: int, limit: int):
    result = await session.execute(
        select(MindUser.mind_user_id)
        .order_by(MindUser.id)
        .offset(offset)
        .limit(limit)
    )
    return result.scalars().all()

async def process_batch(user_ids: list[str]):
    async with AsyncSessionLocal() as session:
        tasks = []
        for uid in user_ids:
            tasks.append(build_user_profiles(session, uid))
        
        # Run concurrently
        await asyncio.gather(*tasks)
        await session.commit()

async def main(limit: int = 1000, batch_size: int = 100):
    logger.info(f"Starting batch profile generation (limit={limit}, batch_size={batch_size})")
    
    start_time = time.time()
    
    async with AsyncSessionLocal() as session:
        total_users = await session.scalar(select(func.count(MindUser.id)))
    
    limit = min(limit, total_users) if limit > 0 else total_users
    logger.info(f"Processing {limit} users out of {total_users} total users...")
    
    processed = 0
    while processed < limit:
        current_batch_size = min(batch_size, limit - processed)
        
        async with AsyncSessionLocal() as session:
            user_ids = await get_user_batch(session, processed, current_batch_size)
            
        if not user_ids:
            break
            
        await process_batch(user_ids)
        
        processed += len(user_ids)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(f"Processed {processed}/{limit} users. ({rate:.2f} users/sec)")
        
    logger.info(f"Done! Processed {processed} users in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Number of users to process (0 for all)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    args = parser.parse_args()
    
    asyncio.run(main(limit=args.limit, batch_size=args.batch_size))
