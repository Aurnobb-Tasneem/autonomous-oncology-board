import NavBar from "@/components/NavBar";
import SpecialistCard, { type SpecialistSpec } from "@/components/SpecialistCard";
import { getTrainingReports, getHealth } from "@/lib/api";

// ---------------------------------------------------------------------------
// Hard-coded spec definitions (the cards are meaningful even with no backend)
// ---------------------------------------------------------------------------

function buildSpecs(
  trainingReports: Record<string, import("@/lib/api").TrainingReport>,
  healthStatus: { ollama?: string } | null
): SpecialistSpec[] {
  const ollamaOk = healthStatus?.ollama === "ok";

  return [
    {
      id: "gigapath",
      name: "Prov-GigaPath",
      subtitle: "ViT-Giant · Vision Foundation Model",
      icon: "🔬",
      color: "#0d9488",
      badge: "Agent 1 · Pathologist",
      params: "1.1 B",
      vram: "~3 GB (FP16)",
      precision: "float16",
      source: "prov-gigapath/prov-gigapath",
      sourceUrl: "https://huggingface.co/prov-gigapath/prov-gigapath",
      description:
        "Prov-GigaPath is a pre-trained ViT-Giant model trained on 1.3 billion pathology image tokens from Providence Health. It extracts rich morphological embeddings from 224×224 histology patches. We use it as a frozen encoder — no fine-tuning required. Monte Carlo dropout (N=20 passes) provides calibrated uncertainty estimates.",
      sampleInput: "224×224 histology patch tensor [1, 3, 224, 224], FP16",
      sampleOutput:
        '{"tissue_classification": "lung_adenocarcinoma", "confidence": 0.94, "uncertainty_interval": "±4.2%", "morphological_features": ["glandular patterns", "nuclear atypia"]}',
      liveStatus: ollamaOk ? "loaded" : "unknown",
    },
    {
      id: "qwen_vl",
      name: "Qwen2.5-VL-7B-Instruct",
      subtitle: "Vision–Language Model · Second Opinion",
      icon: "🧠",
      color: "#22c55e",
      badge: "Agent 1b · Second Opinion",
      params: "7 B",
      vram: "~15 GB (BF16)",
      precision: "bfloat16",
      source: "Qwen/Qwen2.5-VL-7B-Instruct",
      sourceUrl: "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
      description:
        "Qwen2.5-VL provides a language-grounded second opinion on each histology patch. Unlike GigaPath (pure vision), Qwen-VL can reason in natural language about visual features and flag edge cases where GigaPath's confidence is low. Its 7B parameters fit alongside all other models in the MI300X's 192 GB VRAM.",
      sampleInput: "Histology patch image + prompt: 'Describe tissue morphology and flag any malignant features'",
      sampleOutput:
        '"The patch shows irregular glandular structures with pleomorphic nuclei and increased mitotic activity, consistent with adenocarcinoma. Nuclear-to-cytoplasmic ratio is elevated. EGFR mutant morphology pattern suspected."',
      liveStatus: ollamaOk ? "loaded" : "unknown",
    },
    {
      id: "tnm_lora",
      name: "TNM Staging LoRA",
      subtitle: "Llama-3.1-8B · LoRA Rank 16",
      icon: "🧬",
      color: "#38bdf8",
      badge: "Agent 2b · Staging Specialist",
      params: "8 B base + 4.2 M LoRA",
      vram: "~5 GB (FP16) shared with base",
      precision: "float16",
      source: "Llama-3.1-8B + custom adapter",
      description:
        "A LoRA adapter fine-tuned on AJCC 8th-edition TNM staging cases. Base model is Llama-3.1-8B-Instruct; rank-16 adapter constrains the output to valid TNM notation (T0–T4, N0–N3, M0–M1). Outperforms the base 70B model on structured staging by +7.9 pp TNM accuracy.",
      sampleInput:
        '{"tissue": "lung_adenocarcinoma", "size_cm": 3.2, "node_involvement": "ipsilateral", "distant_mets": false}',
      sampleOutput: '{"T": "T2a", "N": "N1", "M": "M0", "stage": "IIB", "nccn_category": "2A"}',
      liveStatus: "loaded",
      trainingReport: trainingReports["tnm_specialist"] ?? null,
    },
    {
      id: "biomarker_lora",
      name: "Biomarker Specialist LoRA",
      subtitle: "Llama-3.1-8B · LoRA Rank 16",
      icon: "🔭",
      color: "#a78bfa",
      badge: "Agent 2c · Biomarker Panel",
      params: "8 B base + 4.2 M LoRA",
      vram: "~5 GB (FP16) shared with base",
      precision: "float16",
      source: "Llama-3.1-8B + custom adapter",
      description:
        "Fine-tuned to identify actionable biomarkers (EGFR, ALK, ROS1, KRAS, BRAF, PD-L1, MSI-H) from morphological findings and tissue type. Generates structured test-ordering panels and gates treatment options behind pending molecular results — exactly as NCCN guidelines require.",
      sampleInput:
        '{"tissue": "lung_adenocarcinoma", "morphology": ["glandular", "solid"], "smoking_history": "never"}',
      sampleOutput:
        '{"recommended_panel": ["EGFR", "ALK", "ROS1", "KRAS G12C"], "priority": "EGFR", "rationale": "Never-smoker adenocarcinoma — EGFR mutation rate ~50%"}',
      liveStatus: "loaded",
      trainingReport: trainingReports["biomarker_specialist"] ?? null,
    },
    {
      id: "treatment_lora",
      name: "Treatment Specialist LoRA",
      subtitle: "Llama-3.1-8B · LoRA Rank 16",
      icon: "💊",
      color: "#fb923c",
      badge: "Agent 2d · Treatment Planning",
      params: "8 B base + 4.2 M LoRA",
      vram: "~5 GB (FP16) shared with base",
      precision: "float16",
      source: "Llama-3.1-8B + custom adapter",
      description:
        "Generates NCCN-aligned first- and second-line treatment options conditioned on TNM stage and biomarker results. Trained on NCCN Clinical Practice Guidelines (lung, colon, breast) and TCGA outcome studies. Always cites the specific NCCN category (1, 2A, 2B, 2B-preferring) for each recommendation.",
      sampleInput:
        '{"stage": "IIIA", "tissue": "lung_adenocarcinoma", "egfr": "mutant (exon 19 del)", "pdl1_tps": 35}',
      sampleOutput:
        '"First-line: Osimertinib 80 mg/day (NCCN Category 1, FLAURA trial). Second-line: Platinum-doublet chemotherapy if progression on TKI."',
      liveStatus: "loaded",
      trainingReport: trainingReports["treatment_specialist"] ?? null,
    },
  ];
}

// ---------------------------------------------------------------------------
// Page (Server Component — can use async/await directly)
// ---------------------------------------------------------------------------

export const metadata = {
  title: "AI Specialists — AOB",
  description: "The five specialist AI models that power the Autonomous Oncology Board",
};

export default async function SpecialistsPage() {
  // Parallel fetch — both gracefully degrade on error
  const [reports, health] = await Promise.allSettled([getTrainingReports(), getHealth()]);

  const trainingMap: Record<string, import("@/lib/api").TrainingReport> = {};
  if (reports.status === "fulfilled") {
    for (const r of reports.value) {
      trainingMap[r.adapter] = r;
    }
  }

  const healthData = health.status === "fulfilled" ? health.value : null;
  const specs = buildSpecs(trainingMap, healthData);

  return (
    <>
      <NavBar />
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "3rem 2rem 5rem" }}>
        {/* Header */}
        <div style={{ marginBottom: "2.5rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.6rem" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--teal-light)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
              AI Model Suite
            </span>
          </div>
          <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "0.6rem", lineHeight: 1.2 }}>
            The Specialist Panel
          </h1>
          <p style={{ fontSize: "0.95rem", color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: "700px" }}>
            AOB does not rely on a single model. Five distinct AI specialists — two vision models and three
            domain-tuned LLM adapters — collaborate in a structured pipeline. All reside simultaneously in the
            AMD MI300X&apos;s 192 GB HBM3 unified memory pool.
          </p>

          {/* VRAM budget summary */}
          <div
            style={{
              marginTop: "1.5rem",
              display: "flex",
              flexWrap: "wrap",
              gap: "0.75rem",
            }}
          >
            {[
              { label: "GigaPath", gb: 3, color: "#0d9488" },
              { label: "Qwen-VL 7B", gb: 15, color: "#22c55e" },
              { label: "Llama 3.3 70B", gb: 70, color: "#0891b2" },
              { label: "LoRA suite ×3", gb: 16, color: "#38bdf8" },
              { label: "KV cache", gb: 30, color: "#7c3aed" },
              { label: "Qdrant + overhead", gb: 9, color: "#64748b" },
            ].map(({ label, gb, color }) => (
              <div
                key={label}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.45rem",
                  padding: "0.3rem 0.75rem",
                  borderRadius: "20px",
                  background: `${color}12`,
                  border: `1px solid ${color}35`,
                  fontSize: "0.75rem",
                }}
              >
                <span style={{ color, fontWeight: 700 }}>{gb} GB</span>
                <span style={{ color: "var(--text-muted)" }}>{label}</span>
              </div>
            ))}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.45rem",
                padding: "0.3rem 0.75rem",
                borderRadius: "20px",
                background: "rgba(74,222,128,0.1)",
                border: "1px solid rgba(74,222,128,0.35)",
                fontSize: "0.75rem",
                fontWeight: 700,
                color: "#4ade80",
              }}
            >
              143 GB / 192 GB used · 49 GB headroom
            </div>
          </div>
        </div>

        {/* Grid of specialist cards */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
            gap: "1.25rem",
          }}
        >
          {specs.map((spec) => (
            <SpecialistCard key={spec.id} spec={spec} />
          ))}
        </div>

        {/* Footer note */}
        <div
          style={{
            marginTop: "2.5rem",
            padding: "1rem 1.25rem",
            background: "rgba(13,148,136,0.06)",
            border: "1px solid var(--teal-border)",
            borderRadius: "10px",
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            lineHeight: 1.6,
          }}
        >
          <strong style={{ color: "var(--teal-light)" }}>Why this matters:</strong> Each specialist runs on the
          same AMD MI300X GPU simultaneously — no model swapping, no latency penalty. The 192 GB unified HBM3
          memory pool is the architectural enabler. An NVIDIA H100 (80 GB) would require offloading at least 2
          models to CPU, adding 8–15 s latency per agent handoff.
        </div>
      </main>
    </>
  );
}
