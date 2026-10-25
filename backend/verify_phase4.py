"""
Phase 4 Verification Script
Computes drift for 3 real users, measures performance, and prints results.
"""

import asyncio
import time
from app.database.connection import AsyncSessionLocal
from app.services.interest_modeling.drift_service import get_drift_report
from app.services.interest_modeling.profile_builder import build_user_profiles

async def verify():
    test_users = ["U536528", "U13740", "U91836"]
    
    async with AsyncSessionLocal() as session:
        print("=== PHASE 4 FINAL RESULTS ===")
        print("Drift metric: Jensen-Shannon Divergence (JSD)")
        print("Feature dimensions: Categories and Subcategories (aligned union)")
        print("Evidence model: Recent timestamped clicks (ADEQUATE>=10, LIMITED>=5)")
        print(f"\nUsers tested: {len(test_users)}\n")
        
        for i, uid in enumerate(test_users):
            t0 = time.perf_counter()
            drift = await get_drift_report(session, uid)
            if not drift:
                await build_user_profiles(session, uid)
                drift = await get_drift_report(session, uid)
                
            t1 = time.perf_counter()
            
            print(f"User {i+1}: {uid}")
            if not drift:
                print("No profile available (no clicks).\n")
                continue
                
            print(f"Historical clicks: {drift['evidence']['historical_clicks']}")
            print(f"Timestamped clicks: {drift['evidence']['timestamped_clicks']}")
            print(f"Recent timestamped clicks: {drift['evidence']['recent_timestamped_clicks']}")
            print(f"JSD: {drift['jsd_score']:.4f}")
            print(f"Evidence: {drift['evidence_level']}")
            
            emerging = [f"{c['feature']} (+{c['delta']:.3f})" for c in drift['emerging'][:3]]
            declining = [f"{c['feature']} ({c['delta']:.3f})" for c in drift['declining'][:3]]
            
            print(f"Emerging: {', '.join(emerging) if emerging else 'None'}")
            print(f"Declining: {', '.join(declining) if declining else 'None'}")
            print(f"(Compute Time: {(t1 - t0)*1000:.2f} ms)")
            print("-" * 40)

if __name__ == "__main__":
    asyncio.run(verify())
