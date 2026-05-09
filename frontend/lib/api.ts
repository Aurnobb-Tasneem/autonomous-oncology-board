// lib/api.ts — Typed API client for AOB backend

/**
 * All API calls go to the Next.js route `app/api/proxy/[...path]/route.ts`, which forwards
 * to FastAPI using `BACKEND_INTERNAL_URL` or `NEXT_PUBLIC_API_URL` at **request time** (so
 * `next build` + `next start` still hits the right host after you edit `.env.local`).
 */
export function getApiBase(): string {
  return "/api/proxy";
}

// ── Types ──────────────────────────────────────────────────────────────────

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface AnalyzeResponse {
  job_id: string;
  case_id: string;
  status: JobStatus;
  message: string;
}

export interface VramProcessRow {
  process: string;
  label: string;
  gb: number;
}

export interface VramModelComponent {
  id: string;
  label: string;
  gb: number;
}

export interface VramInfo {
  used_gb: number;
  total_gb: number;
  free_gb: number;
  percent_used: number;
  /** GiB (1024^3) from VRAM byte counters */
  used_gib?: number;
  total_gib?: number;
  free_gib?: number;
  percent_gib?: number;
  model_breakdown?: {
    gigapath_gb: number;
    qwen_vl_gb?: number;
    llama_gb: number;
    llama_8b_gb?: number;
    lora_gb?: number;
    tnm_lora_gb?: number;
    kv_cache_gb: number;
    runtime_overhead_gb?: number | null;
    [key: string]: number | null | undefined;
  };
  /** Friendly rows from backend (rocm-smi --showpids + labels) */
  processes_display?: VramProcessRow[] | null;
  /** Estimated GigaPath / Llama weights / KV / overhead for dashboard copy */
  model_components?: VramModelComponent[] | null;
  unattributed_gpu_gb?: number | null;
  ollama_model?: string;
  /** Real per-process VRAM from rocm-smi --showpids  e.g. { uvicorn: 20.8, ollama: 48.2 } */
  processes?: Record<string, number> | null;
  source: "rocm-smi" | "mock";
}

export interface DemoCase {
  case_name: string;
  tissue_type: string;
  description: string;
  metadata: Record<string, unknown>;
}

export interface PatchFinding {
  patch_id: number;
  tissue_class: string;
  class_confidence: number;
  abnormality_score: number;
  embedding_norm: number;
}

export interface PathologyReport {
  case_id: string;
  n_patches: number;
  tissue_type: string;
  confidence: number;
  patch_findings: PatchFinding[];
  summary: string;
  flags: string[];
  processing_time_s: number;
  heatmaps_b64: string[];
  uncertainty_interval: string;
  uncertainty_std: number;
  high_uncertainty: boolean;
  biomarkers: Record<string, { score: number; level: string }>;
}

export interface TreatmentOption {
  line: string;
  regimen: string;
  evidence_level: string;
  citation: string;
}

export interface ResearchSummary {
  case_id: string;
  tissue_type: string;
  key_findings: string[];
  recommended_tests: string[];
  treatment_options: TreatmentOption[];
  biomarker_requirements: { biomarker: string; status: string; action: string }[];
  citations: string[];
  evidence_quality: string;
}

export interface ManagementPlan {
  case_id: string;
  generated_at: string;
  patient_summary: string;
  diagnosis: { primary: string; tnm_stage: string; confidence: number };
  immediate_actions: string[];
  treatment_plan: { first_line: string; rationale: string; alternatives: string[] };
  further_investigations: string[];
  multidisciplinary_referrals: string[];
  follow_up: string;
  confidence_score: number;
  board_consensus: string;
  disclaimer: string;
  citations: string[];
  // Digital Twin
  pfs_12mo: number;
  pfs_curve: { month: number; pfs: number }[];
  // Debate
  debate_transcript: DebateRound[];
  revision_notes: string;
  consensus_score: number;
}

export interface DebateRound {
  round: number;
  /**
   * Legacy round-structured fields (kept for backwards compatibility).
   */
  researcher_challenge?: string;
  pathologist_referee?: string;
  oncologist_revision?: string;
  consensus_score?: number;
  revision_notes?: string;

  /**
   * New debate transcript message shape emitted by the board.
   * Example: { round: 1, speaker: "researcher", message: "..." }
   */
  speaker?: string;
  message?: string;
  revised_first_line?: string;
}

export interface BoardResult {
  job_id: string;
  case_id: string;
  status: JobStatus;
  pathology_report?: PathologyReport;
  research_summary?: ResearchSummary;
  management_plan?: ManagementPlan;
  heatmaps_b64?: string[];
  debate_rounds?: DebateRound[];
  similar_cases?: SimilarCase[];
  total_time_s?: number;
  error?: string;
  /** True when one or more LoRA specialists fell back to built-in heuristics. */
  degraded_mode?: boolean;
  /** Names of specialists that were unavailable: "tnm" | "biomarker" | "treatment" */
  unavailable_specialists?: string[];
  /** Core agents that fell back to rule-based heuristics (LLM unavailable / parse failed). */
  fallback_agents?: string[];
}

export interface SpecialistsHealth {
  status: "ok" | "degraded" | "unreachable";
  endpoint: string;
  tnm: boolean;
  biomarker: boolean;
  treatment: boolean;
  models?: string[];
  error?: string;
}

export interface SimilarCase {
  case_id: string;
  tissue_type: string;
  similarity: number;
  first_line_tx: string;
  plan_summary: string;
}

export type AgentId =
  | "pathologist"
  | "second_opinion"
  | "researcher"
  | "tnm_specialist"
  | "oncologist"
  | "debate"
  | "system";

export interface SseStep {
  type: string;
  message: string;
  data?: Record<string, unknown>;
  step?: number;
  total_steps?: number;
  /**
   * Optional agent identifier emitted by the backend SSE stream.
   * Kept permissive to avoid breaking on new/unknown agents.
   */
  agent?: AgentId | (string & {});
  /** Optional progress percent from backend AgentStep (0–100). */
  progress?: number;
  timestamp?: string;
}

// ── API Functions ──────────────────────────────────────────────────────────

export async function getHealth(): Promise<{ status: string; ollama: string; specialists_status?: string }> {
  const res = await fetch(`${getApiBase()}/health`, { cache: "no-store" });
  return res.json();
}

export async function getSpecialistsHealth(): Promise<SpecialistsHealth> {
  const res = await fetch(`${getApiBase()}/health/specialists`, { cache: "no-store" });
  return res.json();
}

export async function getVram(): Promise<VramInfo> {
  const res = await fetch(`${getApiBase()}/api/vram`, { cache: "no-store" });
  if (!res.ok) throw new Error(`VRAM ${res.status}`);
  return res.json();
}

export interface VramHistoryResponse {
  points: {
    ts: number;
    used_gb: number;
    total_gb: number;
    used_gib?: number;
    total_gib?: number;
    pct: number;
  }[];
  current_gb: number;
  total_gb: number;
  current_gib?: number;
  total_gib?: number;
  h100_limit_gb: number;
  mi300x_total_gb: number;
  mi300x_total_gib?: number;
  oom_if_h100: boolean;
}

export async function getVramHistory(seconds: number): Promise<VramHistoryResponse> {
  const res = await fetch(`${getApiBase()}/api/vram/history?seconds=${seconds}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`VRAM history ${res.status}`);
  return res.json();
}

export async function getDemoCases(): Promise<DemoCase[]> {
  const res = await fetch(`${getApiBase()}/demo/cases`, { cache: "no-store" });
  const data = await res.json();
  return data.cases ?? [];
}

export async function runDemoCase(caseName: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${getApiBase()}/demo/run/${encodeURIComponent(caseName)}`, {
    method: "POST",
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Demo run failed: ${res.status}`);
  return res.json();
}

/** Full job snapshot from GET /status/{job_id} (works through Next proxy; SSE often does not). */
export interface JobStatusPayload {
  job_id: string;
  case_id: string;
  status: JobStatus;
  steps: Array<{
    agent: string;
    message: string;
    timestamp: string;
    progress: number;
  }>;
  created_at: string;
  error?: string | null;
}

export async function getJobStatus(jobId: string): Promise<JobStatusPayload> {
  const res = await fetch(`${getApiBase()}/status/${encodeURIComponent(jobId)}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Status ${res.status}`);
  return res.json();
}

export async function analyzeImages(
  images: File[],
  metadata?: Record<string, unknown>
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  images.forEach((img) => formData.append("files", img));
  if (metadata) formData.append("metadata", JSON.stringify(metadata));
  const res = await fetch(`${getApiBase()}/analyze`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
  return res.json();
}

/** One shot — use when you implement your own polling. */
export async function fetchReportOnce(
  jobId: string
): Promise<
  | { ok: true; data: BoardResult }
  | { ok: false; pending: true; message?: string }
  | { ok: false; pending: false; error: string }
> {
  const base = getApiBase();
  const id = encodeURIComponent(jobId);
  const res = await fetch(`${base}/report/${id}`, { cache: "no-store" });
  if (res.status === 202) {
    const j = (await res.json().catch(() => ({}))) as { status?: string; message?: string };
    return { ok: false, pending: true, message: j.message ?? "Still processing" };
  }
  if (res.status === 404) {
    let detail =
      "Job not found — check the URL or run the case again (the API may have restarted and lost in-memory jobs).";
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null)
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* ignore */
    }
    return { ok: false, pending: false, error: detail };
  }
  if (res.status === 500) {
    let detail = "Pipeline failed";
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (j.detail != null) detail = String(j.detail);
    } catch {
      /* ignore */
    }
    return { ok: false, pending: false, error: detail };
  }
  if (!res.ok) {
    let extra = "";
    try {
      const j = (await res.json()) as { detail?: unknown; backend?: string; path?: string };
      if (j.detail != null) extra = `: ${j.detail}`;
      else if (j.backend) extra = ` (proxy → ${j.backend}/${j.path ?? ""})`;
    } catch {
      /* ignore */
    }
    return { ok: false, pending: false, error: `Report fetch failed: ${res.status}${extra}` };
  }
  const data = (await res.json()) as BoardResult;
  return { ok: true, data };
}

/**
 * GET /report/{job_id} — polls while the backend returns 202 (still running).
 */
export async function getReport(
  jobId: string,
  opts?: { pollMs?: number; maxWaitMs?: number }
): Promise<BoardResult> {
  const pollMs = opts?.pollMs ?? 1500;
  const maxWaitMs = opts?.maxWaitMs ?? 600_000;
  const deadline = Date.now() + maxWaitMs;
  for (;;) {
    const r = await fetchReportOnce(jobId);
    if (r.ok) return r.data;
    if (r.pending) {
      if (Date.now() > deadline) {
        throw new Error("Report still processing — refresh this page in a moment.");
      }
      await new Promise((resolve) => setTimeout(resolve, pollMs));
      continue;
    }
    throw new Error(r.error);
  }
}

export async function getHeatmaps(jobId: string): Promise<{ heatmaps: string[]; pending: boolean }> {
  const res = await fetch(`${getApiBase()}/heatmaps/${encodeURIComponent(jobId)}`, { cache: "no-store" });
  if (res.status === 202) return { heatmaps: [], pending: true };
  if (!res.ok) return { heatmaps: [], pending: false };
  const data = await res.json();
  // Backend returns heatmaps_b64; normalize for the UI.
  const heatmaps: string[] = data.heatmaps_b64 ?? data.heatmaps ?? [];
  return { heatmaps, pending: false };
}

export async function getMemoryCases(): Promise<SimilarCase[]> {
  const res = await fetch(`${getApiBase()}/memory/cases`, { cache: "no-store" });
  const data = await res.json();
  return data.cases ?? [];
}

export function streamJob(
  jobId: string,
  onStep: (step: SseStep) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  const es = new EventSource(`${getApiBase()}/stream/${encodeURIComponent(jobId)}`);
  let finished = false;

  const handleDone = () => {
    if (finished) return;
    finished = true;
    onDone();
    es.close();
  };

  const handleError = (msg: string) => {
    if (finished) return;
    finished = true;
    onError(msg);
    es.close();
  };

  /** FastAPI /stream/{job_id} emits AgentStep JSON on the default message event. */
  es.onmessage = (e) => {
    try {
      const raw = JSON.parse(e.data) as Record<string, unknown>;

      // Legacy/alternate shapes (keep tolerant)
      if (raw.type === "done") {
        handleDone();
        return;
      }
      if (raw.type === "error") {
        handleError(String(raw.message ?? "Pipeline error"));
        return;
      }

      const agent = typeof raw.agent === "string" ? raw.agent : undefined;
      const message = typeof raw.message === "string" ? raw.message : "";
      const progress = typeof raw.progress === "number" ? raw.progress : undefined;
      const timestamp = typeof raw.timestamp === "string" ? raw.timestamp : undefined;

      const step: SseStep = {
        type: agent ?? "step",
        message,
        agent,
        progress,
        timestamp,
      };
      onStep(step);
    } catch {
      // ignore parse errors
    }
  };

  /** Same endpoint also emits: `event: done` with `{status, job_id}` — this MUST be handled
   *  or the browser will raise `onerror` when the stream closes (shows as "Connection lost"). */
  es.addEventListener("done", () => {
    handleDone();
  });

  es.onerror = () => {
    // EventSource errors also fire on normal close in some browsers; don't overwrite a clean finish.
    if (finished) return;
    handleError("Connection lost");
  };

  return () => es.close();
}

// ── New endpoint types ─────────────────────────────────────────────────────

export interface TrainingReport {
  adapter: string;
  base_model: string;
  rank: number;
  alpha: number;
  train_loss: number;
  eval_loss?: number;
  best_epoch?: number;
  total_steps: number;
  learning_rate: number;
  batch_size: number;
  finished_at?: string;
}

export interface ConcurrentRunRequest {
  cases: string[];
}

export interface ConcurrentRunResponse {
  run_id: string;
  job_ids: string[];
  cases: string[];
  started_at: string;
}

export interface CounterfactualRequest {
  hypothesis: string;
}

export interface CounterfactualResponse {
  job_id: string;
  hypothesis: string;
  original_first_line: string;
  revised_plan: Partial<ManagementPlan>;
  diff_summary: string;
}

// ── New API Functions ───────────────────────────────────────────────────────

export async function getTrainingReports(): Promise<TrainingReport[]> {
  try {
    const res = await fetch(`${getApiBase()}/api/training/reports`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.reports ?? data ?? [];
  } catch {
    return [];
  }
}

export async function runConcurrent(cases: string[]): Promise<ConcurrentRunResponse> {
  const res = await fetch(`${getApiBase()}/api/concurrent/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cases }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Concurrent run failed: ${res.status}`);
  return res.json();
}

export async function runCounterfactual(
  jobId: string,
  hypothesis: string
): Promise<CounterfactualResponse> {
  const res = await fetch(`${getApiBase()}/api/counterfactual/${encodeURIComponent(jobId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hypothesis }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Counterfactual failed: ${res.status}`);
  return res.json();
}

// ── Helpers ────────────────────────────────────────────────────────────────

export function tissueLabel(t: string): string {
  return t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatConfidence(c: number): string {
  return `${(c * 100).toFixed(1)}%`;
}
