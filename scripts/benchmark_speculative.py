"""
scripts/benchmark_speculative.py
==================================
Benchmark speculative decoding speedup: Llama 3.3 70B with 8B draft
vs standard Llama 3.3 70B, measured on 20 oncology synthesis prompts.

Runs against the vLLM server at :8007 (speculative) and :11434 (Ollama/standard).
Outputs a JSON report to aob/eval/results/speculative_benchmark.json.

Usage:
    # 1. Start standard Ollama (if not running):
    ollama serve

    # 2. Start speculative vLLM server:
    bash scripts/serve_speculative.sh &

    # 3. Run benchmark:
    python scripts/benchmark_speculative.py

    # Quick mode (5 prompts):
    python scripts/benchmark_speculative.py --n_prompts 5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark_speculative")

# 20 representative oncology synthesis prompts
BENCHMARK_PROMPTS = [
    "Synthesise a patient management plan for a 64-year-old male with EGFR exon 19 deletion lung adenocarcinoma, Stage IV with liver metastases, ECOG PS 1.",
    "Provide a comprehensive treatment recommendation for HER2-positive breast carcinoma Stage IIB, post-neoadjuvant pertuzumab + trastuzumab, residual disease found.",
    "Generate a multidisciplinary tumour board summary for colon adenocarcinoma pT4b N2b M1a, BRAF V600E mutant, MSS, liver-only metastatic disease.",
    "Outline first-line treatment options for ALK-rearranged non-small cell lung cancer Stage IIIB, ECOG PS 0, no brain metastases.",
    "Create a clinical management plan for MSI-H metastatic colorectal cancer, first-line therapy naive, BRAF wild-type, KRAS wild-type, left-sided primary.",
    "Discuss treatment escalation for lung adenocarcinoma with osimertinib progression, T790M negative, MET amplification confirmed on rebiopsy.",
    "Prepare a patient management plan for stage IV lung squamous cell carcinoma, PD-L1 TPS 75%, ECOG PS 1, no prior systemic therapy.",
    "Synthesise recommendations for KRAS G12C-mutant lung adenocarcinoma after failure of carboplatin + pemetrexed + pembrolizumab.",
    "Generate treatment options for ROS1 fusion-positive NSCLC with two brain metastases, treatment-naive, age 42.",
    "Outline adjuvant therapy options for resected EGFR L858R lung adenocarcinoma Stage IIB, margins clear, 4/16 nodes positive.",
    "Create management plan for colon adenocarcinoma Lynch syndrome confirmed (MLH1 germline), pT3 N2a M1b (lung + liver), MSI-H.",
    "Summarise first-line treatment for metastatic pancreatic ductal adenocarcinoma, BRCA2 germline mutation, ECOG PS 1.",
    "Provide tumour board recommendations for HER2 IHC 3+ metastatic gastric adenocarcinoma after first-line FOLFOX + trastuzumab progression.",
    "Outline treatment for BRAF V600E-mutant Stage IV melanoma with 3 brain metastases, ECOG PS 0, no prior targeted therapy.",
    "Generate management plan for urothelial carcinoma Stage IVA, FGFR3 alteration confirmed, cisplatin-ineligible patient.",
    "Synthesise a Patient Management Plan for triple-negative breast cancer Stage III, PDL-1 CPS ≥10, post-neoadjuvant carboplatin + taxane.",
    "Describe treatment options for NRG1 fusion-positive lung adenocarcinoma, treatment-naive, Stage IV, age 55.",
    "Create management for endometrial carcinoma Stage IVB, MSI-H dMMR, clear cell histology, ECOG PS 2.",
    "Provide recommendations for MET exon 14 skipping-mutant lung adenocarcinoma, first-line, Stage IV, ECOG PS 1.",
    "Summarise second-line treatment for EGFR exon 20 insertion NSCLC after carboplatin + pemetrexed failure, amivantamab not yet received.",
]


def measure_throughput(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    timeout: int = 90,
) -> Optional[dict]:
    """
    Send one completion request and return token throughput metrics.

    Returns dict with:
        prompt_len:     approximate prompt tokens
        output_tokens:  tokens generated
        latency_s:      wall-clock seconds
        tokens_per_sec: output tokens / latency
        ttft_s:         time-to-first-token (streaming)
    """
    system_msg = "You are a senior oncologist. Generate a structured clinical response."

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  max_tokens,
        "temperature": 0.0,
        "stream":      False,
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            resp = client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        latency = time.perf_counter() - t0
        data    = resp.json()

        usage   = data.get("usage", {})
        out_tok = usage.get("completion_tokens", max_tokens)
        prm_tok = usage.get("prompt_tokens", len(prompt.split()))

        return {
            "prompt_len":    prm_tok,
            "output_tokens": out_tok,
            "latency_s":     round(latency, 3),
            "tokens_per_sec": round(out_tok / max(latency, 0.001), 1),
        }
    except Exception as e:
        log.warning(f"Request failed: {e}")
        return None


def run_benchmark(
    base_url: str,
    model: str,
    label: str,
    prompts: list[str],
    max_tokens: int = 256,
    warmup: int = 2,
) -> dict:
    """
    Run the benchmark against one vLLM endpoint.

    Args:
        base_url:   e.g. "http://localhost:8007/v1"
        model:      model name string
        label:      human-readable name ("speculative" or "standard")
        prompts:    list of prompt strings
        max_tokens: max generation tokens per prompt
        warmup:     number of warmup requests to discard

    Returns:
        dict with summary statistics.
    """
    log.info(f"[{label}] Warming up ({warmup} requests)...")
    for p in prompts[:warmup]:
        measure_throughput(base_url, model, p, max_tokens)

    log.info(f"[{label}] Running {len(prompts)} benchmark prompts...")
    results = []
    for i, prompt in enumerate(prompts):
        r = measure_throughput(base_url, model, prompt, max_tokens)
        if r:
            results.append(r)
            log.info(
                f"  [{label}] {i+1:2d}/{len(prompts)}  "
                f"tokens={r['output_tokens']}  "
                f"latency={r['latency_s']}s  "
                f"throughput={r['tokens_per_sec']} tok/s"
            )

    if not results:
        return {"label": label, "error": "No successful results", "n": 0}

    tps_list = [r["tokens_per_sec"] for r in results]
    lat_list = [r["latency_s"] for r in results]

    return {
        "label":          label,
        "n":              len(results),
        "tokens_per_sec": {
            "mean":    round(statistics.mean(tps_list), 1),
            "median":  round(statistics.median(tps_list), 1),
            "stdev":   round(statistics.stdev(tps_list) if len(tps_list) > 1 else 0, 1),
            "min":     round(min(tps_list), 1),
            "max":     round(max(tps_list), 1),
        },
        "latency_s": {
            "mean":    round(statistics.mean(lat_list), 2),
            "median":  round(statistics.median(lat_list), 2),
            "p90":     round(sorted(lat_list)[int(0.9 * len(lat_list))], 2),
        },
        "raw": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark speculative decoding vs standard vLLM.")
    p.add_argument("--speculative_url",  default="http://localhost:8007/v1")
    p.add_argument("--speculative_model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--standard_url",     default="http://localhost:11434/v1")
    p.add_argument("--standard_model",   default="llama3.3:70b")
    p.add_argument("--n_prompts",        type=int, default=20)
    p.add_argument("--max_tokens",       type=int, default=256)
    p.add_argument("--output",           default="aob/eval/results/speculative_benchmark.json")
    return p.parse_args()


def main():
    args   = parse_args()
    prompts = BENCHMARK_PROMPTS[:args.n_prompts]

    log.info("=" * 60)
    log.info("  Speculative Decoding Benchmark")
    log.info("  AMD MI300X · Llama 3.3 70B + 8B Draft")
    log.info("=" * 60)

    results = {}

    # Speculative decoding
    log.info(f"\nBenchmarking SPECULATIVE server at {args.speculative_url} ...")
    results["speculative"] = run_benchmark(
        base_url=args.speculative_url,
        model=args.speculative_model,
        label="speculative_70B+8B_draft",
        prompts=prompts,
        max_tokens=args.max_tokens,
    )

    # Standard (Ollama)
    log.info(f"\nBenchmarking STANDARD server at {args.standard_url} ...")
    results["standard"] = run_benchmark(
        base_url=args.standard_url,
        model=args.standard_model,
        label="standard_70B_ollama",
        prompts=prompts,
        max_tokens=args.max_tokens,
    )

    # Compute speedup
    spec_tps = results["speculative"].get("tokens_per_sec", {}).get("mean", 0)
    std_tps  = results["standard"].get("tokens_per_sec", {}).get("mean", 0)
    speedup  = round(spec_tps / std_tps, 2) if std_tps > 0 else None

    report = {
        "hardware": "AMD Instinct MI300X 192GB HBM3",
        "main_model": "Llama-3.3-70B-Instruct",
        "draft_model": "Llama-3.1-8B-Instruct",
        "num_speculative_tokens": 5,
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "results": results,
        "speedup_factor": speedup,
        "summary": (
            f"Speculative decoding: {spec_tps:.1f} tok/s | "
            f"Standard: {std_tps:.1f} tok/s | "
            f"Speedup: {speedup}x"
        ) if speedup else "Benchmark incomplete",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("  Benchmark Results")
    log.info("=" * 60)
    if speedup:
        log.info(f"  Speculative: {spec_tps:.1f} tok/s")
        log.info(f"  Standard:    {std_tps:.1f} tok/s")
        log.info(f"  Speedup:     {speedup}x")
    log.info(f"  Saved to:    {out_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
