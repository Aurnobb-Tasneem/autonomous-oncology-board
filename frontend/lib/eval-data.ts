/**
 * 3-step fallback ladder for benchmark / evaluation data.
 *
 * Step 1 — live backend  (/api/proxy/api/benchmark/latest)
 * Step 2 — static JSON   (/eval-results/*.json shipped with the app)
 * Step 3 — hard-coded    constants from the technical report
 *
 * Guarantees the UI renders correctly on Vercel, localhost without ROCm,
 * and on the live MI300X — same code path, no dead screens.
 */

const BASE = "/api/proxy";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AblationRow {
  config: string;
  config_short: string;
  tnm_accuracy: { mean: number; ci_low: number; ci_high: number };
  biomarker_f1: { mean: number; ci_low: number; ci_high: number };
  tx_alignment: { mean: number; ci_low: number; ci_high: number };
  schema_validity: { mean: number; ci_low: number; ci_high: number };
  delta_tnm: number;
  is_full: boolean;
}

export interface AblationData {
  generated_at?: string;
  n_cases: number;
  ablation_table: AblationRow[];
}

export interface ReliabilityPoint {
  bin_mid: number;
  fraction_positive: number;
}

export interface CalibrationModelData {
  ece: number;
  mce: number;
  brier: number;
  reliability_curve: ReliabilityPoint[];
}

export interface CalibrationData {
  generated_at?: string;
  n_cases: number;
  gigapath: CalibrationModelData;
  board_consensus: CalibrationModelData;
}

export interface ClinicalEvalSummary {
  tnm_accuracy: number;
  biomarker_f1: number;
  tx_alignment: number;
  schema_validity: number;
  mean_debate_rounds: number;
  mean_consensus_score: number;
  mean_inference_time_s: number;
}

export interface ClinicalEvalData {
  dataset: string;
  version: string;
  n_cases: number;
  summary: ClinicalEvalSummary;
  per_cancer_type: Record<string, { n: number; tnm_accuracy: number; biomarker_f1: number }>;
  hf_dataset?: string;
  hf_url?: string;
}

// ---------------------------------------------------------------------------
// Fallback constants (from technical_report.md)
// ---------------------------------------------------------------------------

const FALLBACK_ABLATION: AblationData = {
  n_cases: 100,
  ablation_table: [
    { config: "AOB Full Pipeline", config_short: "Full", tnm_accuracy: { mean: 0.823, ci_low: 0.805, ci_high: 0.844 }, biomarker_f1: { mean: 0.748, ci_low: 0.729, ci_high: 0.769 }, tx_alignment: { mean: 0.778, ci_low: 0.761, ci_high: 0.795 }, schema_validity: { mean: 0.970, ci_low: 0.960, ci_high: 0.980 }, delta_tnm: 0.0, is_full: true },
    { config: "No Debate Rounds", config_short: "No Debate", tnm_accuracy: { mean: 0.771, ci_low: 0.752, ci_high: 0.791 }, biomarker_f1: { mean: 0.701, ci_low: 0.680, ci_high: 0.722 }, tx_alignment: { mean: 0.734, ci_low: 0.715, ci_high: 0.753 }, schema_validity: { mean: 0.960, ci_low: 0.948, ci_high: 0.972 }, delta_tnm: -0.052, is_full: false },
    { config: "No LoRA Specialists", config_short: "No LoRA", tnm_accuracy: { mean: 0.744, ci_low: 0.723, ci_high: 0.765 }, biomarker_f1: { mean: 0.678, ci_low: 0.656, ci_high: 0.700 }, tx_alignment: { mean: 0.711, ci_low: 0.691, ci_high: 0.731 }, schema_validity: { mean: 0.950, ci_low: 0.937, ci_high: 0.963 }, delta_tnm: -0.079, is_full: false },
    { config: "No Qwen-VL Second Opinion", config_short: "No Qwen-VL", tnm_accuracy: { mean: 0.798, ci_low: 0.778, ci_high: 0.818 }, biomarker_f1: { mean: 0.724, ci_low: 0.703, ci_high: 0.745 }, tx_alignment: { mean: 0.756, ci_low: 0.737, ci_high: 0.775 }, schema_validity: { mean: 0.968, ci_low: 0.957, ci_high: 0.979 }, delta_tnm: -0.025, is_full: false },
    { config: "Single LLM Baseline", config_short: "Baseline", tnm_accuracy: { mean: 0.691, ci_low: 0.668, ci_high: 0.714 }, biomarker_f1: { mean: 0.612, ci_low: 0.589, ci_high: 0.635 }, tx_alignment: { mean: 0.644, ci_low: 0.622, ci_high: 0.666 }, schema_validity: { mean: 0.920, ci_low: 0.904, ci_high: 0.936 }, delta_tnm: -0.132, is_full: false },
  ],
};

const FALLBACK_CALIBRATION: CalibrationData = {
  n_cases: 100,
  gigapath: { ece: 0.0886, mce: 0.1432, brier: 0.1621, reliability_curve: [{ bin_mid: 0.05, fraction_positive: 0.031 }, { bin_mid: 0.15, fraction_positive: 0.148 }, { bin_mid: 0.25, fraction_positive: 0.219 }, { bin_mid: 0.35, fraction_positive: 0.322 }, { bin_mid: 0.45, fraction_positive: 0.438 }, { bin_mid: 0.55, fraction_positive: 0.541 }, { bin_mid: 0.65, fraction_positive: 0.628 }, { bin_mid: 0.75, fraction_positive: 0.747 }, { bin_mid: 0.85, fraction_positive: 0.821 }, { bin_mid: 0.95, fraction_positive: 0.916 }] },
  board_consensus: { ece: 0.0723, mce: 0.1218, brier: 0.1387, reliability_curve: [{ bin_mid: 0.05, fraction_positive: 0.038 }, { bin_mid: 0.15, fraction_positive: 0.159 }, { bin_mid: 0.25, fraction_positive: 0.237 }, { bin_mid: 0.35, fraction_positive: 0.341 }, { bin_mid: 0.45, fraction_positive: 0.451 }, { bin_mid: 0.55, fraction_positive: 0.558 }, { bin_mid: 0.65, fraction_positive: 0.641 }, { bin_mid: 0.75, fraction_positive: 0.758 }, { bin_mid: 0.85, fraction_positive: 0.839 }, { bin_mid: 0.95, fraction_positive: 0.934 }] },
};

const FALLBACK_CLINICAL_EVAL: ClinicalEvalData = {
  dataset: "AOB-Bench ClinicalEval v1",
  version: "1.0.0",
  n_cases: 100,
  summary: { tnm_accuracy: 0.823, biomarker_f1: 0.748, tx_alignment: 0.778, schema_validity: 0.970, mean_debate_rounds: 1.4, mean_consensus_score: 82.3, mean_inference_time_s: 47.2 },
  per_cancer_type: {
    lung_adenocarcinoma: { n: 30, tnm_accuracy: 0.867, biomarker_f1: 0.812 },
    colon_adenocarcinoma: { n: 25, tnm_accuracy: 0.840, biomarker_f1: 0.756 },
    lung_squamous: { n: 20, tnm_accuracy: 0.800, biomarker_f1: 0.720 },
    breast_idc: { n: 15, tnm_accuracy: 0.800, biomarker_f1: 0.693 },
    other: { n: 10, tnm_accuracy: 0.700, biomarker_f1: 0.620 },
  },
  hf_dataset: "aob-bench/ClinicalEval",
  hf_url: "https://huggingface.co/datasets/aob-bench/ClinicalEval",
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function tryFetch<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Public loaders — each goes live API → static JSON → hardcoded constant
// ---------------------------------------------------------------------------

export async function loadAblationData(): Promise<AblationData> {
  const live = await tryFetch<AblationData>(`${BASE}/api/benchmark/ablation`);
  if (live?.ablation_table?.length) return live;

  const staticData = await tryFetch<AblationData>("/eval-results/ablation.json");
  if (staticData?.ablation_table?.length) return staticData;

  return FALLBACK_ABLATION;
}

export async function loadCalibrationData(): Promise<CalibrationData> {
  const live = await tryFetch<CalibrationData>(`${BASE}/api/benchmark/calibration`);
  if (live?.gigapath) return live;

  const staticData = await tryFetch<CalibrationData>("/eval-results/calibration.json");
  if (staticData?.gigapath) return staticData;

  return FALLBACK_CALIBRATION;
}

export async function loadClinicalEvalData(): Promise<ClinicalEvalData> {
  const live = await tryFetch<ClinicalEvalData>(`${BASE}/api/benchmark/latest`);
  if (live?.summary) return live;

  const staticData = await tryFetch<ClinicalEvalData>("/eval-results/clinical_eval.json");
  if (staticData?.summary) return staticData;

  return FALLBACK_CLINICAL_EVAL;
}
