/**
 * mock-fixtures.ts
 *
 * Canonical mock data for the AOB demo mode (Vercel hosted, no MI300X required).
 * Used by the Next.js proxy route when DEMO_MODE=mock.
 *
 * Job IDs in mock mode are encoded as "mock-{startEpochMs}-{slug}" so each
 * GET /status and GET /report call can compute elapsed time without server state.
 */

// ── Stage definitions ──────────────────────────────────────────────────────────

/** One pipeline step, timed from job start. */
export interface MockStageDef {
  elapsed_ms: number;
  agent: string;
  message: string;
  progress: number;
}

export const MOCK_STAGE_DEFS: MockStageDef[] = [
  {
    elapsed_ms: 0,
    agent: "system",
    message: "Board session initialised — AMD MI300X 192 GB HBM3 unified memory",
    progress: 0,
  },
  {
    elapsed_ms: 1200,
    agent: "pathologist",
    message: "Loading Prov-GigaPath ViT-Giant encoder (FP16, ~3 GB VRAM)",
    progress: 5,
  },
  {
    elapsed_ms: 3500,
    agent: "pathologist",
    message: "Encoding 12 patches — MC Dropout ×20 stochastic forward passes",
    progress: 15,
  },
  {
    elapsed_ms: 8000,
    agent: "pathologist",
    message:
      "Tissue classified: Lung Adenocarcinoma — 94.2% ± 3.1% (uncertainty: low)",
    progress: 28,
  },
  {
    elapsed_ms: 9200,
    agent: "researcher",
    message: "Querying Qdrant in-process corpus — retrieving top-8 chunks",
    progress: 35,
  },
  {
    elapsed_ms: 14500,
    agent: "researcher",
    message:
      "Evidence loaded: NCCN NSCLC 2024 guidelines + 7 TCGA studies — synthesising via Llama 3.3 70B",
    progress: 50,
  },
  {
    elapsed_ms: 17000,
    agent: "tnm_specialist",
    message: "TNM staging: T2bN2M0 → Stage IIIA confirmed (AJCC 8th Ed.)",
    progress: 62,
  },
  {
    elapsed_ms: 21500,
    agent: "oncologist",
    message: "Synthesising Patient Management Plan — Llama 3.3 70B generating (~40 GB VRAM)",
    progress: 75,
  },
  {
    elapsed_ms: 26000,
    agent: "debate",
    message:
      "⚠️ Researcher challenge: EGFR status unknown — NCCN Category 1 TKI therapy requires molecular confirmation first",
    progress: 85,
  },
  {
    elapsed_ms: 30500,
    agent: "debate",
    message:
      "Oncologist revision accepted: molecular panel added to immediate actions — consensus improving",
    progress: 92,
  },
  {
    elapsed_ms: 33000,
    agent: "system",
    message: "Board consensus 87/100 — plan finalised and signed off (1 debate round)",
    progress: 100,
  },
];

/** Total ms until the mock pipeline is considered complete. */
export const MOCK_TOTAL_MS = MOCK_STAGE_DEFS[MOCK_STAGE_DEFS.length - 1].elapsed_ms + 1000;

// ── Job ID helpers ─────────────────────────────────────────────────────────────

export function makeMockJobId(slug = "lung_adenocarcinoma"): string {
  return `mock-${Date.now()}-${slug.replace(/[^a-z0-9_]/gi, "_")}`;
}

export function parseMockJobId(jobId: string): { startMs: number; slug: string } | null {
  const m = jobId.match(/^mock-(\d+)-(.+)$/);
  if (!m) return null;
  return { startMs: parseInt(m[1], 10), slug: m[2] };
}

export function isMockJobId(jobId: string): boolean {
  return jobId.startsWith("mock-");
}

// ── Status computation ─────────────────────────────────────────────────────────

export function getMockStatus(jobId: string): {
  job_id: string;
  case_id: string;
  status: "queued" | "running" | "done" | "failed";
  steps: { agent: string; message: string; timestamp: string; progress: number }[];
  created_at: string;
  error: null;
} {
  const parsed = parseMockJobId(jobId);
  const startMs = parsed?.startMs ?? Date.now() - MOCK_TOTAL_MS;
  const slug = parsed?.slug ?? "lung_adenocarcinoma";
  const elapsed = Date.now() - startMs;

  const visibleSteps = MOCK_STAGE_DEFS.filter((s) => s.elapsed_ms <= elapsed);
  const isDone = elapsed >= MOCK_TOTAL_MS;

  return {
    job_id: jobId,
    case_id: `demo_${slug}`,
    status: isDone ? "done" : elapsed > 0 ? "running" : "queued",
    steps: visibleSteps.map((s) => ({
      agent: s.agent,
      message: s.message,
      timestamp: new Date(startMs + s.elapsed_ms).toISOString(),
      progress: s.progress,
    })),
    created_at: new Date(startMs).toISOString(),
    error: null,
  };
}

// ── Full board result ──────────────────────────────────────────────────────────

export function getMockReport(jobId: string): object | null {
  const parsed = parseMockJobId(jobId);
  const startMs = parsed?.startMs ?? 0;
  const elapsed = Date.now() - startMs;

  if (elapsed < MOCK_TOTAL_MS) return null; // still running

  const slug = parsed?.slug ?? "lung_adenocarcinoma";
  const caseId = `demo_${slug}`;
  const generatedAt = new Date(startMs + MOCK_TOTAL_MS).toISOString();

  return {
    job_id: jobId,
    case_id: caseId,
    status: "done",
    total_time_s: 34.2,
    degraded_mode: false,
    unavailable_specialists: [],
    fallback_agents: [],

    pathology_report: {
      case_id: caseId,
      n_patches: 12,
      tissue_type: "lung_adenocarcinoma",
      confidence: 0.942,
      patch_findings: [
        { patch_id: 0, tissue_class: "lung_adenocarcinoma", class_confidence: 0.96, abnormality_score: 0.88, embedding_norm: 14.3 },
        { patch_id: 1, tissue_class: "lung_adenocarcinoma", class_confidence: 0.91, abnormality_score: 0.81, embedding_norm: 13.7 },
        { patch_id: 2, tissue_class: "lung_adenocarcinoma", class_confidence: 0.94, abnormality_score: 0.85, embedding_norm: 14.1 },
        { patch_id: 3, tissue_class: "normal_lung",         class_confidence: 0.78, abnormality_score: 0.12, embedding_norm: 11.2 },
        { patch_id: 4, tissue_class: "lung_adenocarcinoma", class_confidence: 0.97, abnormality_score: 0.93, embedding_norm: 15.0 },
      ],
      summary:
        "12 patches analysed. Predominant pattern: acinar and lepidic growth with nuclear atypia and irregular glandular structures consistent with lung adenocarcinoma. 9/12 patches flagged as suspicious. MC Dropout uncertainty: low (±3.1%).",
      flags: ["glandular_pattern", "nuclear_atypia", "lepidic_growth", "high_grade_nuclei"],
      processing_time_s: 8.1,
      heatmaps_b64: [],
      uncertainty_interval: "94.2% ± 3.1%",
      uncertainty_std: 0.031,
      high_uncertainty: false,
      biomarkers: {
        EGFR: { score: 0.72, level: "high" },
        ALK:  { score: 0.18, level: "low" },
        PD_L1:{ score: 0.55, level: "moderate" },
        KRAS: { score: 0.22, level: "low" },
        ROS1: { score: 0.11, level: "low" },
      },
    },

    research_summary: {
      case_id: caseId,
      tissue_type: "lung_adenocarcinoma",
      key_findings: [
        "EGFR mutations occur in ~30% of lung adenocarcinomas; molecular testing is mandatory before first-line therapy (NCCN 2024)",
        "PD-L1 ≥50% qualifies for pembrolizumab monotherapy (NCCN Category 1, KEYNOTE-024)",
        "Stage IIIA disease may be resectable; multidisciplinary review required for surgical candidacy",
        "TCGA LUAD data confirms high mutational heterogeneity — comprehensive NGS panel recommended",
      ],
      recommended_tests: [
        "EGFR mutation analysis (exons 18–21)",
        "ALK/ROS1 FISH or IHC",
        "PD-L1 IHC (22C3 clone)",
        "Comprehensive NGS panel (KRAS, BRAF, MET, RET, ERBB2)",
        "Staging CT chest/abdomen/pelvis + brain MRI",
      ],
      treatment_options: [
        {
          line: "1st",
          regimen: "Osimertinib 80 mg OD",
          evidence_level: "NCCN Category 1",
          citation: "FLAURA2: Wu YL et al. NEJM 2023 — EGFR-mutant NSCLC",
        },
        {
          line: "1st",
          regimen: "Pembrolizumab monotherapy (PD-L1 ≥50%)",
          evidence_level: "NCCN Category 1",
          citation: "KEYNOTE-024: Reck M et al. NEJM 2018",
        },
        {
          line: "2nd",
          regimen: "Carboplatin + Pemetrexed + Pembrolizumab",
          evidence_level: "NCCN Category 2A",
          citation: "KEYNOTE-189: Gandhi L et al. NEJM 2018",
        },
        {
          line: "2nd",
          regimen: "Docetaxel + Ramucirumab",
          evidence_level: "NCCN Category 2A",
          citation: "REVEL: Garon EB et al. Lancet 2014",
        },
      ],
      biomarker_requirements: [
        {
          biomarker: "EGFR",
          status: "unknown",
          action: "Order NGS panel before initiating TKI therapy — result gates first-line choice",
        },
        {
          biomarker: "PD-L1",
          status: "unknown",
          action: "IHC 22C3 required — result determines pembrolizumab eligibility",
        },
        {
          biomarker: "ALK",
          status: "unknown",
          action: "FISH or IHC mandatory per NCCN — ALK+ → alectinib first-line",
        },
      ],
      citations: [
        "Wu YL et al. FLAURA2: Osimertinib + chemotherapy vs osimertinib alone. NEJM 2023;389:645–657",
        "Reck M et al. KEYNOTE-024: Pembrolizumab vs platinum chemotherapy. NEJM 2018;379:2040–2051",
        "Gandhi L et al. KEYNOTE-189: Pembrolizumab + pemetrexed + carboplatin. NEJM 2018;378:2078–2092",
        "Garon EB et al. REVEL: Ramucirumab + docetaxel. Lancet 2014;384:665–673",
        "NCCN Clinical Practice Guidelines in Oncology: Non-Small Cell Lung Cancer v4.2024",
      ],
      evidence_quality: "high",
    },

    management_plan: {
      case_id: caseId,
      generated_at: generatedAt,
      patient_summary:
        "65-year-old patient presenting with a peripheral right upper lobe mass (3.2 cm) with ipsilateral mediastinal lymphadenopathy. Non-smoker. Histopathology confirms lung adenocarcinoma (acinar/lepidic pattern, Grade 2). Biomarker status pending. Performance status ECOG 1.",
      diagnosis: {
        primary: "Lung Adenocarcinoma (Acinar/Lepidic pattern, Grade 2)",
        tnm_stage: "T2bN2M0 — Stage IIIA (AJCC 8th Ed.)",
        confidence: 0.942,
      },
      immediate_actions: [
        "Order comprehensive molecular panel: EGFR (exons 18–21), ALK FISH, ROS1 IHC, PD-L1 22C3, KRAS G12C, BRAF, MET, RET, ERBB2, TMB, MSI",
        "Obtain staging CT chest/abdomen/pelvis with contrast",
        "Brain MRI with gadolinium (baseline for stage IIIA)",
        "Multidisciplinary tumour board referral for surgical candidacy assessment",
        "Pulmonary function tests (FEV1, DLCO) if surgery being considered",
      ],
      treatment_plan: {
        first_line:
          "PENDING MOLECULAR RESULTS — do not initiate targeted therapy before biomarker results are available. If EGFR+: Osimertinib 80 mg OD (NCCN Category 1). If PD-L1 ≥50% & driver-negative: Pembrolizumab 200 mg Q3W (NCCN Category 1). If driver-negative, PD-L1 <50%: Carboplatin AUC5 + Pemetrexed 500 mg/m² + Pembrolizumab 200 mg Q3W × 4 cycles (NCCN Category 2A).",
        rationale:
          "NCCN 2024 mandates comprehensive molecular testing before first-line systemic therapy in adenocarcinoma. TKI therapy in EGFR-mutant disease yields superior PFS vs chemotherapy (FLAURA2). Initiating chemotherapy before results risks suboptimal sequencing and precludes eligibility for category 1 targeted options.",
        alternatives: [
          "Clinical trial enrollment (preferred per NCCN if available at institution)",
          "Definitive chemoradiation (Stage IIIA unresectable): Cisplatin + Etoposide + concurrent RT → Durvalumab consolidation (PACIFIC regimen)",
          "Surgical resection + adjuvant osimertinib if EGFR+ and R0 resection achieved (ADAURA data)",
        ],
      },
      further_investigations: [
        "PET-CT scan for accurate mediastinal staging",
        "EBUS-TBNA or mediastinoscopy for pathological N2 confirmation",
        "Echocardiogram (baseline cardiotoxicity assessment before platinum therapy)",
        "Bone scan or whole-body PET if bone metastases suspected",
        "Repeat biopsy if molecular panel fails (insufficient tissue)",
      ],
      multidisciplinary_referrals: [
        "Thoracic surgery — surgical candidacy and VATS assessment",
        "Radiation oncology — concurrent chemoRT planning if unresectable",
        "Palliative care — early parallel integration per ASCO guidelines",
        "Clinical genetics — germline testing if patient <50 or non-smoker",
        "Nutritional support — pre-treatment nutritional optimisation",
      ],
      follow_up:
        "Re-convene board within 7 days of molecular results. If driver mutation identified, initiate targeted therapy within 14 days. Response assessment CT at 6–8 weeks post-treatment initiation. If stable/responding, continue and reassess Q8–12 weeks.",
      confidence_score: 87,
      board_consensus:
        "High consensus (87/100) achieved after 1 debate round. Primary revision: treatment initiation deferred pending mandatory molecular testing — consensus improved from 71 to 87 after this amendment. Board aligned on Stage IIIA classification and immediate investigation priorities.",
      disclaimer:
        "This output is an AI research demonstration. NOT a medical device. NOT approved for clinical use. All recommendations must be reviewed and validated by a qualified oncologist before any clinical decision.",
      citations: [
        "Wu YL et al. FLAURA2. NEJM 2023;389:645–657",
        "Reck M et al. KEYNOTE-024. NEJM 2018;379:2040–2051",
        "Gandhi L et al. KEYNOTE-189. NEJM 2018;378:2078–2092",
        "Antonia SJ et al. PACIFIC: Durvalumab after chemoRT. NEJM 2018;379:2342–2350",
        "Wu YL et al. ADAURA: Osimertinib adjuvant. NEJM 2020;383:1711–1723",
        "NCCN NSCLC Guidelines v4.2024",
      ],
      pfs_12mo: 0.78,
      pfs_curve: [
        { month: 0,  pfs: 1.00 },
        { month: 1,  pfs: 0.97 },
        { month: 2,  pfs: 0.95 },
        { month: 3,  pfs: 0.92 },
        { month: 4,  pfs: 0.90 },
        { month: 5,  pfs: 0.88 },
        { month: 6,  pfs: 0.86 },
        { month: 7,  pfs: 0.84 },
        { month: 8,  pfs: 0.82 },
        { month: 9,  pfs: 0.81 },
        { month: 10, pfs: 0.80 },
        { month: 11, pfs: 0.79 },
        { month: 12, pfs: 0.78 },
      ],
      debate_transcript: [
        {
          round: 1,
          speaker: "researcher",
          message:
            "⚠️ CHALLENGE: Treatment plan proposes TKI initiation without confirmed EGFR status. NCCN Category 1 evidence for osimertinib is explicitly conditional on molecular confirmation. Initiating before results violates guideline mandate and risks harm if patient is EGFR-negative.",
          researcher_challenge:
            "EGFR status unknown — NCCN Category 1 TKI therapy requires molecular confirmation before initiation.",
        },
        {
          round: 1,
          speaker: "oncologist",
          message:
            "Revision accepted. Treatment plan amended: first-line therapy explicitly deferred until molecular panel results available. Mandatory molecular testing added to immediate actions with 7-day reconvene.",
          oncologist_revision:
            "Plan revised to mandate molecular testing before any systemic therapy initiation.",
          revised_first_line:
            "PENDING MOLECULAR RESULTS — initiation deferred per NCCN 2024 mandate.",
          consensus_score: 87,
          revision_notes:
            "Initial first-line recommendation updated to explicitly gate on biomarker results.",
        },
      ],
      revision_notes:
        "Round 1: Treatment initiation amended from 'propose TKI' to 'defer pending EGFR confirmation'. Consensus improved from 71 → 87 after revision.",
      consensus_score: 87,
    },

    debate_rounds: [
      {
        round: 1,
        researcher_challenge:
          "EGFR status unknown — NCCN Category 1 TKI therapy requires molecular confirmation before initiation.",
        oncologist_revision:
          "Plan revised to mandate molecular testing before any systemic therapy initiation.",
        consensus_score: 87,
        revision_notes:
          "Treatment initiation deferred pending biomarker confirmation.",
      },
    ],

    similar_cases: [
      {
        case_id: "demo_case_001",
        tissue_type: "lung_adenocarcinoma",
        similarity: 0.93,
        first_line_tx: "Osimertinib 80 mg OD (EGFR exon 19 del confirmed)",
        plan_summary:
          "65F non-smoker, Stage IIIA LUAD, EGFR exon 19 deletion confirmed. Initiated osimertinib — 18-month PFS.",
      },
      {
        case_id: "demo_case_002",
        tissue_type: "lung_adenocarcinoma",
        similarity: 0.87,
        first_line_tx: "Pembrolizumab 200 mg Q3W (PD-L1 78%)",
        plan_summary:
          "58M former smoker, Stage IIIB LUAD, PD-L1 78%, EGFR wild-type. Pembrolizumab monotherapy — 14-month PFS.",
      },
      {
        case_id: "demo_case_003",
        tissue_type: "lung_adenocarcinoma",
        similarity: 0.81,
        first_line_tx: "Carboplatin + Pemetrexed + Pembrolizumab",
        plan_summary:
          "71M never-smoker, Stage IIIA LUAD, all drivers negative, PD-L1 22%. Chemo-immuno combination — 11-month PFS.",
      },
    ],

    heatmaps_b64: [],
  };
}

// ── Demo cases list ────────────────────────────────────────────────────────────

export const MOCK_DEMO_CASES = [
  {
    case_name: "lung_adenocarcinoma",
    tissue_type: "lung_adenocarcinoma",
    description: "Peripheral RUL mass, 3.2 cm, acinar/lepidic pattern. Stage IIIA (T2bN2M0).",
    metadata: { patient_age: 65, sex: "F", smoking_status: "never" },
  },
  {
    case_name: "colon_adenocarcinoma",
    tissue_type: "colon_adenocarcinoma",
    description: "Sigmoid colon adenocarcinoma, moderately differentiated. Stage III (T3N1M0).",
    metadata: { patient_age: 58, sex: "M", smoking_status: "former" },
  },
  {
    case_name: "lung_squamous_cell",
    tissue_type: "lung_squamous_cell_carcinoma",
    description: "Central lung mass, hilar involvement. Squamous cell carcinoma Stage IIIA.",
    metadata: { patient_age: 71, sex: "M", smoking_status: "current" },
  },
];

// ── VRAM info ──────────────────────────────────────────────────────────────────

export const MOCK_VRAM_INFO = {
  used_gb: 88.2,
  total_gb: 191.7,
  free_gb: 103.5,
  percent_used: 46.0,
  used_gib: 82.1,
  total_gib: 178.5,
  free_gib: 96.4,
  percent_gib: 46.0,
  model_breakdown: {
    gigapath_gb: 3.1,
    llama_gb: 40.4,
    kv_cache_gb: 31.2,
    lora_gb: 2.8,
    runtime_overhead_gb: 10.7,
  },
  model_components: [
    { id: "gigapath", label: "Prov-GigaPath ViT-Giant (FP16)", gb: 3.1 },
    { id: "llama",    label: "Llama 3.3 70B (Ollama q4_K_S)",  gb: 40.4 },
    { id: "kv_cache", label: "KV Cache (active sessions)",      gb: 31.2 },
    { id: "lora",     label: "LoRA Adapters (TNM/BM/TX)",       gb: 2.8 },
    { id: "overhead", label: "ROCm runtime + buffers",          gb: 10.7 },
  ],
  unattributed_gpu_gb: null,
  ollama_model: "llama3.3:70b-instruct-q4_K_S",
  processes: { uvicorn: 52.2, ollama: 36.0 },
  processes_display: [
    { process: "uvicorn (FastAPI + GigaPath)", label: "ML API", gb: 52.2 },
    { process: "ollama (Llama 3.3 70B)",       label: "LLM",    gb: 36.0 },
  ],
  source: "mock" as const,
};

export function getMockVramHistory(seconds: number): object {
  const now = Date.now();
  const points = [];
  const n = Math.min(seconds, 120);
  for (let i = n; i >= 0; i--) {
    const t = now - i * 1000;
    // Simulate a ramp that reached steady-state ~90s ago
    const age = i;
    let used: number;
    if (age > 90) {
      used = 5 + (88.2 - 5) * Math.max(0, 1 - (age - 90) / 30);
    } else {
      used = 88.2 + (Math.random() - 0.5) * 0.4;
    }
    used = Math.max(5, Math.min(191.7, used));
    points.push({
      ts: t,
      used_gb: parseFloat(used.toFixed(1)),
      total_gb: 191.7,
      used_gib: parseFloat((used * 0.931).toFixed(1)),
      total_gib: 178.5,
      pct: parseFloat(((used / 191.7) * 100).toFixed(1)),
    });
  }
  return {
    points,
    current_gb: 88.2,
    total_gb: 191.7,
    current_gib: 82.1,
    total_gib: 178.5,
    h100_limit_gb: 80.0,
    mi300x_total_gb: 191.7,
    mi300x_total_gib: 178.5,
    oom_if_h100: true,
  };
}
