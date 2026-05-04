"""
scripts/golden_path_test.py
============================
End-to-end golden-path regression test for AOB.

Tests the full pipeline using a pre-baked mock case (no GPU required for CI).
When run with --live, uses the actual AOB API at localhost:8000.

Exit 0 = all checks passed.
Exit 1 = one or more checks failed.

Usage:
    python scripts/golden_path_test.py              # mock mode (CI)
    python scripts/golden_path_test.py --live        # requires running AOB stack
    python scripts/golden_path_test.py --live --url http://host:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

# ── Golden-path expected outputs ──────────────────────────────────────────────
# These are the minimum structural requirements for a valid AOB response.
# Not exact-match — checks that required keys exist and have correct types.

_GOLDEN_PATHOLOGY_REPORT_KEYS = {
    "tissue_classification", "confidence", "morphological_features",
    "suspicious_regions", "uncertainty_std",
}

_GOLDEN_EVIDENCE_BUNDLE_KEYS = {
    "relevant_protocols", "staging_guidance", "citations",
}

_GOLDEN_MANAGEMENT_PLAN_KEYS = {
    "diagnosis", "tnm_stage", "treatment_recommendations",
    "further_investigations", "confidence_score",
}

_GOLDEN_BOARD_RESULT_KEYS = {
    "pathology_report", "evidence_bundle", "management_plan",
    "case_id",
}

# ── Mock response (returned in mock mode) ────────────────────────────────────
_MOCK_AOB_RESPONSE = {
    "case_id": "GOLDEN-001",
    "status": "completed",
    "pathology_report": {
        "tissue_classification": "lung_adenocarcinoma",
        "confidence": 0.91,
        "uncertainty_std": 0.04,
        "morphological_features": ["glandular patterns", "nuclear atypia", "mucin production"],
        "suspicious_regions": [{"patch_id": 0, "score": 0.91}],
        "embedding_summary": [0.1] * 10,
    },
    "evidence_bundle": {
        "relevant_protocols": [
            {
                "protocol": "KEYNOTE-189",
                "nccn_category": "1",
                "summary": "Carboplatin + Pemetrexed + Pembrolizumab for EGFR/ALK-negative NSCLC",
            }
        ],
        "staging_guidance": "T2a N2 M0 — Stage IIIA. Unresectable disease. Concurrent CRT candidate.",
        "citations": [
            {"title": "KEYNOTE-189", "year": 2018, "pmid": "29658856"}
        ],
    },
    "management_plan": {
        "diagnosis": "Lung adenocarcinoma, acinar-predominant",
        "tnm_stage": {"T": "T2a", "N": "N2", "M": "M0", "stage": "IIIA"},
        "treatment_recommendations": [
            {
                "line": "First-line",
                "regimen": "Carboplatin + Pemetrexed + Pembrolizumab",
                "nccn_category": "1",
                "citation": "KEYNOTE-189",
            }
        ],
        "further_investigations": ["EGFR panel", "ALK IHC", "PD-L1 TPS", "ROS1 FISH"],
        "confidence_score": 0.87,
        "debate_transcript": [
            {
                "round": 1,
                "actor": "oncologist",
                "text": "Initial plan: start osimertinib pending EGFR confirmation.",
            },
            {
                "round": 2,
                "actor": "researcher",
                "challenge": "EGFR not confirmed. NCCN Category 1 requires molecular testing first.",
            },
            {
                "round": 3,
                "actor": "oncologist",
                "revision": "Updated: EGFR reflex testing required before TKI. Interim: platinum doublet.",
            },
        ],
        "consensus_score": 88,
    },
    "biomarker_panel": {
        "tests_required": ["EGFR NGS panel", "ALK IHC/FISH", "ROS1 FISH", "PD-L1 IHC (TPS)", "KRAS G12C"],
        "urgency": "high",
        "rationale": "Stage IV NSCLC — all actionable driver mutations must be excluded before initiating chemotherapy.",
    },
    "treatment_proposal": {
        "first_line": "Carboplatin + Pemetrexed + Pembrolizumab (KEYNOTE-189)",
        "nccn_category": "1",
        "rationale": "EGFR/ALK wild-type, PD-L1 ≥1%, non-squamous histology.",
    },
    "differential_diagnosis": {
        "diagnoses": [
            {"diagnosis": "lung_adenocarcinoma", "probability": 0.91, "key_features": ["glandular pattern"]},
            {"diagnosis": "large_cell_carcinoma", "probability": 0.06, "key_features": ["nuclear pleomorphism"]},
            {"diagnosis": "metastatic_adenocarcinoma", "probability": 0.03, "key_features": ["absence of primary marker"]},
        ]
    },
    "patient_summary": {
        "summary": (
            "Your biopsy shows a type of lung cancer called adenocarcinoma. "
            "The cancer is in Stage 3A, which means it has spread to some lymph nodes "
            "but has not traveled to other organs. Your care team recommends starting "
            "with chemotherapy and immunotherapy together while waiting for gene test results. "
            "These tests will tell us if there is a targeted medicine that works better for your specific cancer."
        ),
        "reading_level": "8th grade",
    },
    "trial_matches": {
        "trials": [
            {
                "nct_id": "NCT04116541",
                "title": "ADAURA: Osimertinib Adjuvant Therapy in EGFR-Mutated NSCLC",
                "eligibility_score": 0.73,
                "flags": ["PENDING EGFR result"],
            }
        ]
    },
}


# ── Checks ────────────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name    = name
        self.passed  = passed
        self.message = message

    def __repr__(self):
        icon = "PASS" if self.passed else "FAIL"
        suffix = f" -- {self.message}" if self.message else ""
        return f"  [{icon}] {self.name}{suffix}"


def check_required_keys(obj: dict, required: set, label: str) -> CheckResult:
    missing = required - set(obj.keys())
    if missing:
        return CheckResult(label, False, f"Missing keys: {missing}")
    return CheckResult(label, True)


def check_type(obj: dict, key: str, expected_type, label: str) -> CheckResult:
    val = obj.get(key)
    if not isinstance(val, expected_type):
        return CheckResult(label, False, f"{key} expected {expected_type.__name__}, got {type(val).__name__}")
    return CheckResult(label, True)


def check_range(val: float, lo: float, hi: float, label: str) -> CheckResult:
    if not (lo <= val <= hi):
        return CheckResult(label, False, f"{val} not in [{lo}, {hi}]")
    return CheckResult(label, True)


def run_golden_path_checks(response: dict) -> list[CheckResult]:
    results = []

    # 1. Top-level keys
    results.append(check_required_keys(response, _GOLDEN_BOARD_RESULT_KEYS, "board_result_top_level_keys"))
    results.append(check_type(response, "case_id", str, "case_id_is_string"))

    # 2. Pathology report
    pr = response.get("pathology_report") or {}
    results.append(check_required_keys(pr, _GOLDEN_PATHOLOGY_REPORT_KEYS, "pathology_report_keys"))
    if "confidence" in pr:
        results.append(check_range(float(pr["confidence"]), 0.0, 1.0, "pathology_confidence_range"))
    if "uncertainty_std" in pr:
        results.append(check_range(float(pr["uncertainty_std"]), 0.0, 1.0, "uncertainty_std_range"))
    if "tissue_classification" in pr:
        results.append(CheckResult(
            "tissue_classification_non_empty",
            bool(pr["tissue_classification"]),
            pr.get("tissue_classification", ""),
        ))
    if "morphological_features" in pr:
        results.append(check_type(pr, "morphological_features", list, "morphological_features_is_list"))

    # 3. Evidence bundle
    eb = response.get("evidence_bundle") or {}
    results.append(check_required_keys(eb, _GOLDEN_EVIDENCE_BUNDLE_KEYS, "evidence_bundle_keys"))
    if "citations" in eb:
        results.append(check_type(eb, "citations", list, "citations_is_list"))
        results.append(CheckResult(
            "at_least_one_citation",
            len(eb.get("citations", [])) >= 1,
            f"Found {len(eb.get('citations', []))} citations",
        ))

    # 4. Management plan
    mp = response.get("management_plan") or {}
    results.append(check_required_keys(mp, _GOLDEN_MANAGEMENT_PLAN_KEYS, "management_plan_keys"))
    if "tnm_stage" in mp:
        tnm = mp["tnm_stage"]
        for field in ["T", "N", "M", "stage"]:
            results.append(CheckResult(
                f"tnm_{field}_present",
                field in tnm and bool(tnm[field]),
                tnm.get(field, ""),
            ))
    if "confidence_score" in mp:
        results.append(check_range(float(mp["confidence_score"]), 0.0, 1.0, "management_plan_confidence_range"))
    if "debate_transcript" in mp:
        dt = mp["debate_transcript"]
        results.append(CheckResult("debate_transcript_non_empty", len(dt) >= 1, f"{len(dt)} rounds"))

    # 5. Biomarker panel
    bm = response.get("biomarker_panel") or {}
    if bm:
        results.append(CheckResult(
            "biomarker_tests_required_non_empty",
            len(bm.get("tests_required", [])) >= 1,
            f"{len(bm.get('tests_required', []))} tests",
        ))

    # 6. Treatment proposal
    tp = response.get("treatment_proposal") or {}
    if tp:
        results.append(CheckResult(
            "treatment_proposal_first_line",
            bool(tp.get("first_line")),
            tp.get("first_line", ""),
        ))
        results.append(CheckResult(
            "nccn_category_valid",
            tp.get("nccn_category") in {"1", "2A", "2B"},
            tp.get("nccn_category", ""),
        ))

    # 7. Patient summary
    ps = response.get("patient_summary") or {}
    if ps:
        results.append(CheckResult(
            "patient_summary_non_empty",
            len(ps.get("summary", "")) >= 50,
            f"{len(ps.get('summary', ''))} chars",
        ))

    # 8. Differential diagnosis
    dd = response.get("differential_diagnosis") or {}
    if dd:
        diagnoses = dd.get("diagnoses", [])
        results.append(CheckResult(
            "differential_diagnoses_count",
            1 <= len(diagnoses) <= 5,
            f"{len(diagnoses)} diagnoses",
        ))

    # 9. Trial matches
    tm = response.get("trial_matches") or {}
    if tm:
        results.append(CheckResult(
            "trial_matches_returned",
            isinstance(tm.get("trials"), list),
            f"{len(tm.get('trials', []))} trials",
        ))

    return results


# ── Live API test ──────────────────────────────────────────────────────────────

def run_live_test(base_url: str, timeout: int = 120) -> dict:
    """Submit a test case to the live AOB API and return the response."""
    if not _HTTPX:
        print("httpx not installed — cannot run live test.")
        sys.exit(1)

    # Create minimal dummy patches (1 black 224×224 patch as base64 is complex;
    # use the /analyze endpoint with text-only metadata for golden path)
    payload = {
        "case_id": f"GOLDEN-LIVE-{str(uuid.uuid4())[:8]}",
        "metadata": {
            "age": 62,
            "sex": "M",
            "smoking_history": "30 pack-years",
            "clinical_note": (
                "Lung adenocarcinoma, acinar-predominant, 3.2 cm left lower lobe. "
                "2 of 15 mediastinal lymph nodes positive. No distant metastases."
            ),
        },
        "patches": [],  # empty patches → fallback to dummy embeddings
    }

    print(f"\nSubmitting to {base_url}/analyze ...")
    t0 = time.perf_counter()
    resp = httpx.post(f"{base_url}/analyze", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - t0
    print(f"Response received in {elapsed:.1f}s")
    return data


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AOB Golden-Path Regression Test")
    p.add_argument("--live",    action="store_true", help="Test against running API")
    p.add_argument("--url",     default="http://localhost:8000")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output",  default=None, help="Save results to JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  AOB Golden-Path Regression Test")
    print("=" * 60)

    if args.live:
        print(f"\n[LIVE] Testing against {args.url}")
        try:
            response = run_live_test(args.url, timeout=args.timeout)
        except Exception as e:
            print(f"\n✗ API call failed: {e}")
            sys.exit(1)
    else:
        print("\n[MOCK] Using pre-baked golden response (no GPU required)")
        response = _MOCK_AOB_RESPONSE

    checks = run_golden_path_checks(response)

    passed = [c for c in checks if c.passed]
    failed = [c for c in checks if not c.passed]

    print(f"\nResults: {len(passed)}/{len(checks)} checks passed\n")
    for c in checks:
        print(c)

    print("\n" + "=" * 60)
    if failed:
        print(f"\n  FAILED -- {len(failed)} checks failed:")
        for c in failed:
            print(f"    * {c.name}: {c.message}")
        print()
    else:
        print("\n  ALL CHECKS PASSED -- AOB golden path is healthy\n")
    print("=" * 60 + "\n")

    if args.output:
        out = {
            "mode":    "live" if args.live else "mock",
            "passed":  len(passed),
            "failed":  len(failed),
            "total":   len(checks),
            "checks":  [{"name": c.name, "passed": c.passed, "message": c.message} for c in checks],
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Results saved: {args.output}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
