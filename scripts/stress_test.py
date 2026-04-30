"""
scripts/stress_test.py
======================
AOB Stress Test — 10 concurrent API calls, latency + throughput measurement.

Usage (from project root):
    pip install aiohttp
    python scripts/stress_test.py --host http://localhost:8000 --concurrency 10

Output:
    • Cases/min throughput
    • P50 / P95 / P99 latency per case
    • VRAM peak (reads /api/vram at end)
    • Debate rounds triggered per case
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


# ── Result per case ───────────────────────────────────────────────────────────
@dataclass
class CaseResult:
    case_name: str
    job_id: Optional[str] = None
    success: bool = False
    latency_s: float = 0.0
    debate_rounds: int = 0
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
async def submit_demo_case(
    session: aiohttp.ClientSession,
    host: str,
    case_name: str,
) -> CaseResult:
    """Submit a demo case and poll until completion. Returns CaseResult."""
    result = CaseResult(case_name=case_name)
    t0 = time.perf_counter()

    try:
        # POST /demo/run/{case_name}
        async with session.post(f"{host}/demo/run/{case_name}") as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"submit failed {resp.status}: {body[:200]}"
                return result
            data = await resp.json()
            job_id = data.get("job_id")
            if not job_id:
                result.error = "No job_id in response"
                return result
            result.job_id = job_id

        # Poll /report/{job_id} until done or failed
        while True:
            await asyncio.sleep(2.0)
            async with session.get(f"{host}/report/{job_id}") as resp:
                if resp.status == 202:
                    continue  # still processing
                if resp.status == 500:
                    body = await resp.text()
                    result.error = f"pipeline failed: {body[:200]}"
                    result.latency_s = round(time.perf_counter() - t0, 2)
                    return result
                if resp.status == 200:
                    report = await resp.json()
                    result.success = True
                    result.latency_s = round(time.perf_counter() - t0, 2)
                    result.debate_rounds = report.get("debate_rounds_completed", 0)
                    return result
                # Unexpected status
                result.error = f"unexpected status {resp.status}"
                result.latency_s = round(time.perf_counter() - t0, 2)
                return result

    except Exception as exc:
        result.error = str(exc)
        result.latency_s = round(time.perf_counter() - t0, 2)
        return result


async def fetch_vram(session: aiohttp.ClientSession, host: str) -> dict:
    """Fetch VRAM stats from /api/vram."""
    try:
        async with session.get(f"{host}/api/vram", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return {}


async def check_demo_cases(session: aiohttp.ClientSession, host: str) -> list[str]:
    """List available demo case names from /demo/cases."""
    try:
        async with session.get(f"{host}/demo/cases") as resp:
            if resp.status == 200:
                data = await resp.json()
                return [c["case_name"] for c in data.get("cases", [])]
    except Exception as exc:
        print(f"  WARNING: could not list demo cases: {exc}")
    # Fallback to known names
    return ["lung_adenocarcinoma", "colon_adenocarcinoma", "lung_squamous_cell"]


# ── Main stress test ──────────────────────────────────────────────────────────
async def run_stress_test(host: str, concurrency: int) -> None:
    print(f"\n{'='*60}")
    print(f"  AOB STRESS TEST")
    print(f"  Target: {host}")
    print(f"  Concurrency: {concurrency} simultaneous cases")
    print(f"{'='*60}\n")

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout   = aiohttp.ClientTimeout(total=600)  # 10 min max per case

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Check health first
        print("Checking API health...", end=" ", flush=True)
        try:
            async with session.get(f"{host}/health") as resp:
                health = await resp.json()
                print(f"OK — Ollama: {health.get('ollama', '?')}, board_ready: {health.get('board_ready', '?')}")
        except Exception as exc:
            print(f"FAILED ({exc})")
            print("Cannot connect to API. Is it running?")
            return

        # Get demo case names
        demo_cases = await check_demo_cases(session, host)
        print(f"Available demo cases: {demo_cases}\n")

        # Build the task list — repeat/cycle to reach `concurrency` count
        task_names: list[str] = []
        for i in range(concurrency):
            task_names.append(demo_cases[i % len(demo_cases)])

        print(f"Submitting {concurrency} cases concurrently...\n")
        wall_start = time.perf_counter()

        # Run all cases concurrently
        tasks = [
            submit_demo_case(session, host, name)
            for name in task_names
        ]
        results: list[CaseResult] = await asyncio.gather(*tasks)

        wall_elapsed = time.perf_counter() - wall_start

        # Fetch VRAM after all cases complete
        vram = await fetch_vram(session, host)

    # ── Report ────────────────────────────────────────────────────────────────
    successes = [r for r in results if r.success]
    failures  = [r for r in results if not r.success]
    latencies = [r.latency_s for r in successes]

    print(f"\n{'='*60}")
    print(f"  STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Cases submitted  : {concurrency}")
    print(f"  Succeeded        : {len(successes)}")
    print(f"  Failed           : {len(failures)}")
    print(f"  Wall time        : {wall_elapsed:.1f}s")

    if successes:
        cases_per_min = round(len(successes) / wall_elapsed * 60, 2)
        p50 = round(statistics.median(latencies), 2)
        p95 = round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else latencies[0]
        p99 = round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if len(latencies) > 1 else latencies[0]
        total_debate = sum(r.debate_rounds for r in successes)

        print(f"\n  Throughput       : {cases_per_min} cases/min")
        print(f"  Latency P50      : {p50}s")
        print(f"  Latency P95      : {p95}s")
        print(f"  Latency P99      : {p99}s")
        print(f"  Min / Max        : {min(latencies):.2f}s / {max(latencies):.2f}s")
        print(f"  Debate rounds    : {total_debate} total ({total_debate/len(successes):.1f} avg/case)")

    if vram:
        print(f"\n  VRAM (post-test) : {vram.get('used_gb', '?')} GB / {vram.get('total_gb', '?')} GB "
              f"({vram.get('percent', '?')}%)")
        print(f"  Exceeds H100 80G : {vram.get('exceeds_h100', '?')}")

    if failures:
        print(f"\n  FAILURES:")
        for r in failures:
            print(f"    [{r.case_name}] {r.error}")

    print(f"\n{'='*60}\n")

    # Per-case breakdown
    print("  PER-CASE BREAKDOWN:")
    print(f"  {'Case':<28} {'Status':<10} {'Latency':>10}  {'Debate Rounds':>14}")
    print(f"  {'-'*66}")
    for r in results:
        status = "✅ OK" if r.success else "❌ FAIL"
        lat_str = f"{r.latency_s:.2f}s" if r.latency_s else "—"
        rounds_str = str(r.debate_rounds) if r.success else "—"
        print(f"  {r.case_name:<28} {status:<10} {lat_str:>10}  {rounds_str:>14}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="AOB Stress Test — concurrent API load testing")
    parser.add_argument("--host", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent cases")
    args = parser.parse_args()

    asyncio.run(run_stress_test(host=args.host, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
