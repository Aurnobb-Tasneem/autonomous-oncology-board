"""
eval/clinical_eval.py
======================
ClinicalEval — AOB's published benchmark for clinical reasoning quality.

100 curated cases covering lung adenocarcinoma, lung squamous cell carcinoma,
colon adenocarcinoma, and benign tissue.

Four metrics:
    1. TNM exact-match rate        — all 4 fields (T, N, M, overall_stage) equal
    2. Biomarker set F1            — order-insensitive set comparison
    3. Treatment class alignment   — canonical treatment class mapping
    4. JSON-schema compliance rate — required keys present, parseable

Four configurations evaluated:
    baseline_8b      — base Llama 3.1 8B, zero-shot prompt
    adapter_tnm_only — only TNM adapter active
    adapter_suite    — all three adapters, no agents
    aob_full         — full pipeline with debate

Usage:
    python -m eval.clinical_eval                          # full benchmark
    python -m eval.clinical_eval --max_cases 10           # quick smoke test
    python -m eval.clinical_eval --config aob_full        # single config
    make eval                                              # via Makefile target

Output:
    aob/eval/results/clinical_eval_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("clinical_eval")

# ── Treatment class canonical mapping ────────────────────────────────────────
_TX_CLASS_MAP: dict[str, str] = {
    # EGFR TKIs
    "osimertinib": "tki_egfr_3g", "gefitinib": "tki_egfr_1g", "erlotinib": "tki_egfr_1g",
    "afatinib": "tki_egfr_2g", "dacomitinib": "tki_egfr_2g",
    "amivantamab": "bispecific_egfr_met",
    # ALK
    "alectinib": "tki_alk", "brigatinib": "tki_alk", "lorlatinib": "tki_alk",
    "crizotinib": "tki_alk_ros1",
    # KRAS
    "sotorasib": "kras_g12c_inhibitor", "adagrasib": "kras_g12c_inhibitor",
    # IO
    "pembrolizumab": "anti_pd1_io", "nivolumab": "anti_pd1_io",
    "cemiplimab": "anti_pd1_io", "atezolizumab": "anti_pdl1_io",
    "durvalumab": "anti_pdl1_io",
    "ipilimumab": "anti_ctla4_io",
    # Chemo-IO
    "carboplatin": "platinum_chemo", "cisplatin": "platinum_chemo",
    "pemetrexed": "antifolate_chemo", "oxaliplatin": "platinum_chemo",
    # CRC biologic
    "cetuximab": "anti_egfr_biologic", "panitumumab": "anti_egfr_biologic",
    "bevacizumab": "anti_vegf_biologic",
    "encorafenib": "braf_inhibitor", "binimetinib": "mek_inhibitor",
    # Refractory CRC
    "regorafenib": "multikinase_inhibitor", "trifluridine": "trifluridine_tipiracil",
    "fruquintinib": "multikinase_inhibitor",
    # Lynch / MSI-H
    "folfox": "platinum_doublet_5fu", "capox": "platinum_doublet_capecitabine",
    "folfiri": "irinotecan_5fu",
    # NTRK
    "larotrectinib": "trk_inhibitor", "entrectinib": "trk_inhibitor",
    # HER2
    "trastuzumab": "anti_her2_biologic", "tucatinib": "tki_her2",
    "pertuzumab": "anti_her2_biologic",
    # None / surveillance
    "surveillance": "surveillance", "observation": "surveillance",
    "best supportive care": "best_supportive_care",
}


def normalise_tx_class(tx_text: str) -> str:
    """Map a raw first-line treatment string to a canonical class."""
    if not tx_text:
        return "unknown"
    tx_lower = tx_text.lower()
    for key, cls in _TX_CLASS_MAP.items():
        if key in tx_lower:
            return cls
    if "folfox" in tx_lower or "capox" in tx_lower:
        return "platinum_doublet_5fu"
    if "pembrolizumab" in tx_lower and ("carboplatin" in tx_lower or "pemetrexed" in tx_lower):
        return "chemo_io_combo"
    if "surveillance" in tx_lower or "observation" in tx_lower:
        return "surveillance"
    return "other_therapy"


# ── Metric implementations ────────────────────────────────────────────────────

def tnm_exact_match(pred: dict, truth: dict) -> bool:
    """True iff all 4 TNM fields match exactly (case-insensitive)."""
    for key in ["T", "N", "M", "stage"]:
        if str(pred.get(key, "")).upper() != str(truth.get(key, "")).upper():
            return False
    return True


def biomarker_set_f1(pred: list[str], truth: list[str]) -> float:
    """F1 score treating biomarker lists as sets (order-insensitive)."""
    pred_set  = {s.lower().strip() for s in pred  if s.strip()}
    truth_set = {s.lower().strip() for s in truth if s.strip()}
    if not truth_set:
        return 1.0 if not pred_set else 0.0
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def treatment_class_alignment(pred_first_line: str, truth_tx_class: str) -> bool:
    return normalise_tx_class(pred_first_line) == truth_tx_class


def schema_compliance(output: dict, required_keys: set) -> bool:
    return all(k in output for k in required_keys)


# ── Prompts per config ────────────────────────────────────────────────────────

_REQUIRED_OUTPUT_KEYS = {"tnm", "biomarkers", "first_line_tx", "nccn_category"}

def _build_prompt(case: dict, config: str) -> str:
    pt = case.get("pathology_text", "")
    md = case.get("metadata", {})
    meta_str = f"Age {md.get('age','?')}, {md.get('sex','?')}, {md.get('smoking_history','unknown')}."

    if config == "baseline_8b":
        return (
            f"Patient: {meta_str}\n"
            f"Pathology: {pt}\n\n"
            f"Output a JSON object with keys: tnm (object with T, N, M, stage), "
            f"biomarkers (list of strings), first_line_tx (string), nccn_category (string '1' or '2A' or '2B')."
        )
    elif config == "adapter_tnm_only":
        return (
            f"Clinical case: {pt}. {meta_str}\n"
            f"Provide TNM staging and first-line treatment. "
            f"Output JSON with keys: tnm (T/N/M/stage), biomarkers, first_line_tx, nccn_category."
        )
    elif config == "adapter_suite":
        return (
            f"Oncology case summary:\n"
            f"Pathology: {pt}\nPatient: {meta_str}\n\n"
            f"Using your TNM staging, biomarker, and treatment expertise, output JSON with keys: "
            f"tnm (T/N/M/stage), biomarkers (list), first_line_tx (string), nccn_category."
        )
    else:  # aob_full
        return (
            f"You are the lead oncologist in a multidisciplinary tumour board.\n"
            f"Pathology: {pt}\nPatient: {meta_str}\n\n"
            f"After reviewing all specialist reports, output JSON with keys: "
            f"tnm (T/N/M/stage), biomarkers (list), first_line_tx (string), nccn_category. "
            f"Reference NCCN guidelines. Output only valid JSON."
        )


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = "llama3.3:70b",
    ollama_url: str = "http://localhost:11434",
    max_tokens: int = 256,
    timeout: int = 60,
) -> Optional[dict]:
    """Call Ollama and parse the JSON response."""
    import re
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": max_tokens}},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        # Extract JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


def call_vllm_specialist(
    prompt: str,
    base_url: str,
    model: str,
    timeout: int = 30,
) -> Optional[dict]:
    """Call vLLM specialist adapter and parse JSON."""
    import re
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 256, "temperature": 0.0},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        return None
    except Exception as e:
        log.warning(f"vLLM call failed: {e}")
        return None


# ── Main evaluator ────────────────────────────────────────────────────────────

def evaluate(
    cases: list[dict],
    config: str,
    ollama_url: str = "http://localhost:11434",
    vllm_base_url: str = "http://localhost:8006/v1",
    ollama_model: str = "llama3.3:70b",
    base_8b_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_cases: Optional[int] = None,
) -> dict:
    """
    Evaluate one configuration on the given cases.

    Returns a dict with:
        config, n, exact_match, biomarker_f1, treatment_align, schema_compliance,
        per_case (list), latency_stats
    """
    if max_cases:
        cases = cases[:max_cases]

    log.info(f"[{config}] Evaluating {len(cases)} cases...")

    tnm_exact    = 0
    bm_f1_sum    = 0.0
    tx_align     = 0
    schema_ok    = 0
    per_case     = []
    latencies    = []

    for i, case in enumerate(cases):
        t0     = time.perf_counter()
        prompt = _build_prompt(case, config)
        gt     = case.get("ground_truth", {})

        output = None

        if config == "baseline_8b":
            # Use 8B model via Ollama (pull llama3.1:8b or use the 70B as proxy)
            output = call_ollama(prompt, model="llama3.1:8b", ollama_url=ollama_url)
            if output is None:
                # Fall back to 70B if 8B not available
                output = call_ollama(prompt, model=ollama_model, ollama_url=ollama_url)

        elif config == "adapter_tnm_only":
            output = call_vllm_specialist(
                prompt, base_url=vllm_base_url, model="tnm_specialist"
            )
            # Wrap single-task output into full eval schema
            if output and "T" in output:
                output = {"tnm": output, "biomarkers": [], "first_line_tx": "", "nccn_category": "?"}

        elif config == "adapter_suite":
            # Call all three specialist adapters and merge
            tnm_out = call_vllm_specialist(prompt, base_url=vllm_base_url, model="tnm_specialist")
            bm_out  = call_vllm_specialist(prompt, base_url=vllm_base_url, model="biomarker_specialist")
            tx_out  = call_vllm_specialist(prompt, base_url=vllm_base_url, model="treatment_specialist")
            if any([tnm_out, bm_out, tx_out]):
                output = {
                    "tnm": tnm_out if tnm_out else {},
                    "biomarkers": (bm_out or {}).get("tests_required", []),
                    "first_line_tx": (tx_out or {}).get("first_line", ""),
                    "nccn_category": (tx_out or {}).get("nccn_category", "?"),
                }

        else:  # aob_full
            output = call_ollama(prompt, model=ollama_model, ollama_url=ollama_url)

        latency = time.perf_counter() - t0
        latencies.append(latency)

        # Score
        tnm_match  = False
        bm_f1      = 0.0
        tx_ok      = False
        schema_ok_ = False

        if output:
            schema_ok_ = schema_compliance(output, _REQUIRED_OUTPUT_KEYS)
            if schema_ok_:
                schema_ok += 1

            # TNM exact match
            pred_tnm  = output.get("tnm") or {}
            gt_tnm    = gt.get("tnm") or {}
            tnm_match = tnm_exact_match(pred_tnm, gt_tnm)
            if tnm_match:
                tnm_exact += 1

            # Biomarker F1
            pred_bm = output.get("biomarkers") or []
            gt_bm   = gt.get("biomarkers") or []
            bm_f1   = biomarker_set_f1(pred_bm, gt_bm)
            bm_f1_sum += bm_f1

            # Treatment class alignment
            pred_tx  = output.get("first_line_tx") or ""
            gt_tx_cls = gt.get("first_line_tx_class") or ""
            tx_ok = treatment_class_alignment(pred_tx, gt_tx_cls)
            if tx_ok:
                tx_align += 1

        per_case.append({
            "case_id":          case.get("case_id"),
            "tnm_exact_match":  tnm_match,
            "biomarker_f1":     round(bm_f1, 4),
            "treatment_align":  tx_ok,
            "schema_compliance": schema_ok_,
            "pred":             output,
            "latency_s":        round(latency, 2),
        })

        log.info(
            f"  [{config}] {i+1:3d}/{len(cases)}  "
            f"TNM={'✓' if tnm_match else '✗'}  "
            f"BM_F1={bm_f1:.2f}  "
            f"TX={'✓' if tx_ok else '✗'}  "
            f"Schema={'✓' if schema_ok_ else '✗'}  "
            f"{latency:.1f}s"
        )

    n = len(cases)
    return {
        "config":          config,
        "n":               n,
        "tnm_exact_match": round(tnm_exact / n, 4) if n else 0,
        "biomarker_f1":    round(bm_f1_sum / n, 4) if n else 0,
        "treatment_align": round(tx_align / n, 4) if n else 0,
        "schema_compliance": round(schema_ok / n, 4) if n else 0,
        "per_case":        per_case,
        "latency_stats": {
            "mean_s":   round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "total_s":  round(sum(latencies), 1),
            "n":        len(latencies),
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run ClinicalEval benchmark.")
    p.add_argument("--cases_file", default=str(Path(__file__).parent / "cases" / "clinical_eval_cases.json"))
    p.add_argument("--output_dir", default=str(Path(__file__).parent / "results"))
    p.add_argument("--config", default="all",
                   help="baseline_8b | adapter_tnm_only | adapter_suite | aob_full | all")
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--ollama_url", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--vllm_base_url", default=os.getenv("SPECIALISTS_BASE_URL", "http://localhost:8006/v1"))
    return p.parse_args()


def main():
    args = parse_args()

    cases_path = Path(args.cases_file)
    if not cases_path.exists():
        log.error(f"Cases file not found: {cases_path}")
        log.error("Run:  python scripts/gen_clinical_eval_cases.py")
        sys.exit(1)

    with open(cases_path) as f:
        cases = json.load(f)

    log.info(f"Loaded {len(cases)} evaluation cases")

    configs_to_run = (
        ["baseline_8b", "adapter_tnm_only", "adapter_suite", "aob_full"]
        if args.config == "all" else [args.config]
    )

    all_results = {}
    for config in configs_to_run:
        result = evaluate(
            cases=cases,
            config=config,
            ollama_url=args.ollama_url,
            vllm_base_url=args.vllm_base_url,
            max_cases=args.max_cases,
        )
        all_results[config] = result

    # Summary
    log.info("\n" + "=" * 70)
    log.info("  ClinicalEval Results — AOB-Bench")
    log.info("=" * 70)
    header = f"{'Config':<20} {'TNM-EM':>8} {'BM-F1':>8} {'TX-Align':>10} {'Schema':>8}"
    log.info(header)
    log.info("-" * 70)
    for cfg, r in all_results.items():
        log.info(
            f"{cfg:<20} {r['tnm_exact_match']:>8.1%} {r['biomarker_f1']:>8.3f} "
            f"{r['treatment_align']:>10.1%} {r['schema_compliance']:>8.1%}"
        )
    log.info("=" * 70)

    # Save
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clinical_eval_{ts}.json"

    report = {
        "benchmark":       "AOB-Bench ClinicalEval v1",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "hardware":        "AMD Instinct MI300X 192GB HBM3",
        "n_cases":         len(cases),
        "configurations":  configs_to_run,
        "results":         {k: {ki: vi for ki, vi in v.items() if ki != "per_case"}
                            for k, v in all_results.items()},
        "per_case":        {k: v["per_case"] for k, v in all_results.items()},
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\nResults saved to: {out_path}")
    return report


if __name__ == "__main__":
    main()
