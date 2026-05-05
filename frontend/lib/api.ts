// lib/api.ts — Typed API client for AOB backend

// Proxy all requests through Next.js to avoid Mixed Content (HTTPS -> HTTP) and CORS issues.
const BASE = "/api/proxy";

// ── Types ──────────────────────────────────────────────────────────────────

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface AnalyzeResponse {
  job_id: string;
  case_id: string;
  status: JobStatus;
  message: string;
}

export interface VramInfo {
  used_gb: number;
  total_gb: number;
  free_gb: number;
  percent_used: number;
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

export async function getHealth(): Promise<{ status: string; ollama: string }> {
  const res = await fetch(`${BASE}/health`, { cache: "no-store" });
  return res.json();
}

export async function getVram(): Promise<VramInfo> {
  const res = await fetch(`${BASE}/api/vram`, { cache: "no-store" });
  return res.json();
}

export async function getDemoCases(): Promise<DemoCase[]> {
  const res = await fetch(`${BASE}/demo/cases`, { cache: "no-store" });
  const data = await res.json();
  return data.cases ?? [];
}

export async function runDemoCase(caseName: string): Promise<AnalyzeResponse> {
  const res = await fetch(`${BASE}/demo/run/${caseName}`, {
    method: "POST",
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Demo run failed: ${res.status}`);
  return res.json();
}

export async function analyzeImages(
  images: File[],
  metadata?: Record<string, unknown>
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  images.forEach((img) => formData.append("files", img));
  if (metadata) formData.append("metadata", JSON.stringify(metadata));
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
  return res.json();
}

export async function getReport(jobId: string): Promise<BoardResult> {
  const res = await fetch(`${BASE}/report/${jobId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Report fetch failed: ${res.status}`);
  return res.json();
}

export async function getHeatmaps(jobId: string): Promise<{ heatmaps: string[]; pending: boolean }> {
  const res = await fetch(`${BASE}/heatmaps/${jobId}`, { cache: "no-store" });
  if (res.status === 202) return { heatmaps: [], pending: true };
  if (!res.ok) return { heatmaps: [], pending: false };
  const data = await res.json();
  // Backend returns heatmaps_b64; normalize for the UI.
  const heatmaps: string[] = data.heatmaps_b64 ?? data.heatmaps ?? [];
  return { heatmaps, pending: false };
}

export async function getMemoryCases(): Promise<SimilarCase[]> {
  const res = await fetch(`${BASE}/memory/cases`, { cache: "no-store" });
  const data = await res.json();
  return data.cases ?? [];
}

export function streamJob(
  jobId: string,
  onStep: (step: SseStep) => void,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  const es = new EventSource(`${BASE}/stream/${jobId}`);
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
    const res = await fetch(`${BASE}/api/training/reports`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return data.reports ?? data ?? [];
  } catch {
    return [];
  }
}

export async function runConcurrent(cases: string[]): Promise<ConcurrentRunResponse> {
  const res = await fetch(`${BASE}/api/concurrent/run`, {
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
  const res = await fetch(`${BASE}/api/counterfactual/${jobId}`, {
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
