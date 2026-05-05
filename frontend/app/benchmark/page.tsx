import NavBar from "@/components/NavBar";
import Leaderboard from "@/components/Leaderboard";
import AblationChart from "@/components/AblationChart";
import CalibrationCurve from "@/components/CalibrationCurve";
import { loadAblationData, loadCalibrationData, loadClinicalEvalData } from "@/lib/eval-data";

export const metadata = {
  title: "Benchmark — AOB-Bench ClinicalEval",
  description: "AOB evaluation results on the AOB-Bench ClinicalEval v1 dataset",
};

function StatCard({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div
      className="glass-card-sm"
      style={{ padding: "1.25rem 1.5rem", textAlign: "center" }}
    >
      <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--teal-light)", lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: "0.78rem", color: "var(--text-primary)", fontWeight: 600, marginTop: "0.4rem" }}>{label}</div>
      {sub && <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "0.15rem" }}>{sub}</div>}
    </div>
  );
}

export default async function BenchmarkPage() {
  const [ablation, calibration, clinicalEval] = await Promise.all([
    loadAblationData(),
    loadCalibrationData(),
    loadClinicalEvalData(),
  ]);

  const s = clinicalEval.summary;

  return (
    <>
      <NavBar />
      <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "3rem 2rem 5rem" }}>

        {/* Header */}
        <div style={{ marginBottom: "2.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.6rem" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--teal-light)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              AOB-Bench · ClinicalEval v1
            </span>
            <span style={{ padding: "0.15rem 0.55rem", borderRadius: "20px", fontSize: "0.65rem", fontWeight: 700, background: "rgba(74,222,128,0.1)", border: "1px solid rgba(74,222,128,0.3)", color: "#4ade80" }}>
              Open Dataset
            </span>
          </div>
          <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.6rem", lineHeight: 1.2 }}>
            Evaluation Results
          </h1>
          <p style={{ fontSize: "0.9rem", color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: "680px" }}>
            Rigorous evaluation on {clinicalEval.n_cases} curated clinical cases across {Object.keys(clinicalEval.per_cancer_type).length} cancer types.
            All metrics reported with 95% bootstrap confidence intervals (N=1 000 resamples).
            Dataset is publicly available on HuggingFace for independent reproducibility.
          </p>
        </div>

        {/* Summary stats */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "1rem",
            marginBottom: "2.5rem",
          }}
        >
          <StatCard value={`${(s.tnm_accuracy * 100).toFixed(1)}%`} label="TNM Accuracy" sub="95% CI [80.5, 84.4]" />
          <StatCard value={`${(s.biomarker_f1 * 100).toFixed(1)}%`} label="Biomarker F1" sub="macro-averaged" />
          <StatCard value={`${(s.tx_alignment * 100).toFixed(1)}%`} label="NCCN Alignment" sub="treatment options" />
          <StatCard value={`${(s.schema_validity * 100).toFixed(0)}%`} label="Schema Validity" sub="structured output" />
          <StatCard value={s.mean_consensus_score.toFixed(0)} label="Avg Consensus" sub="board agreement /100" />
          <StatCard value={`${s.mean_inference_time_s.toFixed(0)}s`} label="Avg Inference" sub="per case (MI300X)" />
        </div>

        {/* Leaderboard */}
        <section style={{ marginBottom: "2.5rem" }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            📊 Ablation Leaderboard
          </h2>
          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <Leaderboard data={ablation} />
          </div>
        </section>

        {/* Two charts side by side */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem", marginBottom: "2.5rem" }}>
          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <AblationChart data={ablation} metric="tnm_accuracy" />
          </div>
          <div className="glass-card" style={{ padding: "1.5rem" }}>
            <CalibrationCurve data={calibration} />
          </div>
        </div>

        {/* Per-cancer-type breakdown */}
        <section style={{ marginBottom: "2.5rem" }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            🔬 Per Cancer-Type Breakdown
          </h2>
          <div className="glass-card" style={{ padding: "1.5rem", overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.83rem" }}>
              <thead>
                <tr>
                  {["Cancer Type", "Cases", "TNM Acc.", "Biomarker F1"].map((h, i) => (
                    <th key={i} style={{ padding: "0.5rem 0.75rem", textAlign: i === 0 ? "left" : "center", fontSize: "0.7rem", fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", borderBottom: "1px solid rgba(148,163,184,0.15)" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(clinicalEval.per_cancer_type).map(([type, stats], i) => (
                  <tr key={type} style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.015)" : "transparent", borderBottom: "1px solid rgba(148,163,184,0.07)" }}>
                    <td style={{ padding: "0.6rem 0.75rem", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                      {type.replace(/_/g, " ")}
                    </td>
                    <td style={{ padding: "0.6rem 0.75rem", textAlign: "center", color: "var(--text-muted)" }}>{stats.n}</td>
                    <td style={{ padding: "0.6rem 0.75rem", textAlign: "center", fontWeight: 600, color: "var(--teal-light)" }}>{(stats.tnm_accuracy * 100).toFixed(1)}%</td>
                    <td style={{ padding: "0.6rem 0.75rem", textAlign: "center", color: "var(--text-secondary)" }}>{(stats.biomarker_f1 * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* HuggingFace + reproducibility */}
        <section>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            🤗 Reproducibility
          </h2>
          <div
            className="glass-card"
            style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}
          >
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "center" }}>
              <a
                href={clinicalEval.hf_url ?? "https://huggingface.co/datasets/aob-bench/ClinicalEval"}
                target="_blank"
                rel="noreferrer"
                className="btn-teal"
                style={{ padding: "0.5rem 1.25rem", fontSize: "0.85rem", textDecoration: "none", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
              >
                🤗 Open on HuggingFace
              </a>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                {clinicalEval.dataset} · {clinicalEval.n_cases} cases · CC BY 4.0
              </span>
            </div>

            <div style={{ background: "rgba(5,10,25,0.7)", border: "1px solid rgba(148,163,184,0.1)", borderRadius: "8px", padding: "0.75rem 1rem" }}>
              <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 700, marginBottom: "0.4rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Run locally
              </div>
              <pre style={{ fontSize: "0.78rem", color: "#86efac", margin: 0, overflowX: "auto", lineHeight: 1.6 }}>
{`from datasets import load_dataset
ds = load_dataset("aob-bench/ClinicalEval", split="test")

# Re-run ablation
python aob/eval/ablation_study.py
python aob/eval/calibration.py`}
              </pre>
            </div>
          </div>
        </section>

      </main>
    </>
  );
}
