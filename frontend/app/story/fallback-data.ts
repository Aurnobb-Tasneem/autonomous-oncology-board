import { type AblationData } from "@/lib/eval-data";

export const FALLBACK_ABLATION_INLINE: AblationData = {
  n_cases: 100,
  ablation_table: [
    { config: "AOB Full Pipeline", config_short: "Full", tnm_accuracy: { mean: 0.823, ci_low: 0.805, ci_high: 0.844 }, biomarker_f1: { mean: 0.748, ci_low: 0.729, ci_high: 0.769 }, tx_alignment: { mean: 0.778, ci_low: 0.761, ci_high: 0.795 }, schema_validity: { mean: 0.970, ci_low: 0.960, ci_high: 0.980 }, delta_tnm: 0.0, is_full: true },
    { config: "No Debate Rounds", config_short: "No Debate", tnm_accuracy: { mean: 0.771, ci_low: 0.752, ci_high: 0.791 }, biomarker_f1: { mean: 0.701, ci_low: 0.680, ci_high: 0.722 }, tx_alignment: { mean: 0.734, ci_low: 0.715, ci_high: 0.753 }, schema_validity: { mean: 0.960, ci_low: 0.948, ci_high: 0.972 }, delta_tnm: -0.052, is_full: false },
    { config: "No LoRA Specialists", config_short: "No LoRA", tnm_accuracy: { mean: 0.744, ci_low: 0.723, ci_high: 0.765 }, biomarker_f1: { mean: 0.678, ci_low: 0.656, ci_high: 0.700 }, tx_alignment: { mean: 0.711, ci_low: 0.691, ci_high: 0.731 }, schema_validity: { mean: 0.950, ci_low: 0.937, ci_high: 0.963 }, delta_tnm: -0.079, is_full: false },
    { config: "Single LLM Baseline", config_short: "Baseline", tnm_accuracy: { mean: 0.691, ci_low: 0.668, ci_high: 0.714 }, biomarker_f1: { mean: 0.612, ci_low: 0.589, ci_high: 0.635 }, tx_alignment: { mean: 0.644, ci_low: 0.622, ci_high: 0.666 }, schema_validity: { mean: 0.920, ci_low: 0.904, ci_high: 0.936 }, delta_tnm: -0.132, is_full: false },
  ],
};
