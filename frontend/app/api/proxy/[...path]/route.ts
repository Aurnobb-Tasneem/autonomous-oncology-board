import { NextRequest, NextResponse } from "next/server";

import {
  getMockStatus,
  getMockReport,
  getMockVramHistory,
  isMockJobId,
  makeMockJobId,
  MOCK_DEMO_CASES,
  MOCK_VRAM_INFO,
} from "@/lib/mock-fixtures";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// ── Live proxy helpers ─────────────────────────────────────────────────────────

/** Resolve FastAPI base at request time (not at build time). */
function backendBase(): string {
  const u =
    process.env.BACKEND_INTERNAL_URL?.trim() ||
    process.env.NEXT_PUBLIC_API_URL?.trim() ||
    "http://127.0.0.1:8000";
  return u.replace(/\/$/, "");
}

const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
]);

function filterRequestHeaders(incoming: Headers): Headers {
  const out = new Headers();
  incoming.forEach((value, key) => {
    const k = key.toLowerCase();
    if (HOP_BY_HOP.has(k)) return;
    if (k === "host" || k === "content-length") return;
    out.set(key, value);
  });
  return out;
}

function filterResponseHeaders(incoming: Headers): Headers {
  const out = new Headers();
  incoming.forEach((value, key) => {
    const k = key.toLowerCase();
    if (k === "transfer-encoding") return;
    if (HOP_BY_HOP.has(k)) return;
    out.set(key, value);
  });
  return out;
}

// ── Mock handler ───────────────────────────────────────────────────────────────

/**
 * Intercepts all proxy calls when DEMO_MODE=mock.
 * Uses time-encoded job IDs so no server state is needed — fully works on Vercel
 * serverless functions. Job ID format: "mock-{startEpochMs}-{slug}"
 */
async function handleMock(
  segments: string[],
  req: NextRequest
): Promise<NextResponse> {
  const [seg0, seg1, seg2] = segments;
  const method = req.method.toUpperCase();

  // POST /analyze — upload-based case
  if (seg0 === "analyze" && method === "POST") {
    const jobId = makeMockJobId("custom_upload");
    return NextResponse.json({
      job_id: jobId,
      case_id: "demo_custom_upload",
      status: "queued",
      message: "Mock pipeline started — time-based simulation active",
    });
  }

  // GET /demo/cases
  if (seg0 === "demo" && seg1 === "cases" && method === "GET") {
    return NextResponse.json({ cases: MOCK_DEMO_CASES });
  }

  // POST /demo/run/{caseName}
  if (seg0 === "demo" && seg1 === "run" && seg2) {
    const jobId = makeMockJobId(seg2);
    return NextResponse.json({
      job_id: jobId,
      case_id: `demo_${seg2}`,
      status: "queued",
      message: "Mock pipeline started",
    });
  }

  // GET /status/{jobId}
  if (seg0 === "status" && seg1) {
    const jobId = seg1;
    if (!isMockJobId(jobId)) {
      return NextResponse.json({ detail: "Unknown mock job" }, { status: 404 });
    }
    return NextResponse.json(getMockStatus(jobId));
  }

  // GET /report/{jobId}
  if (seg0 === "report" && seg1) {
    const jobId = seg1;
    if (!isMockJobId(jobId)) {
      return NextResponse.json({ detail: "Unknown mock job" }, { status: 404 });
    }
    const report = getMockReport(jobId);
    if (!report) {
      return NextResponse.json(
        { status: "running", message: "Pipeline still processing" },
        { status: 202 }
      );
    }
    return NextResponse.json(report);
  }

  // GET /heatmaps/{jobId}
  if (seg0 === "heatmaps" && seg1) {
    return NextResponse.json({ heatmaps_b64: [], pending: false });
  }

  // GET /health
  if (seg0 === "health" && !seg1) {
    return NextResponse.json({
      status: "ok",
      mode: "mock",
      ollama: "mock (no MI300X in hosted demo)",
      board_ready: true,
      specialists_status: "mock",
      gpu: "AMD Instinct MI300X (192 GB HBM3)",
    });
  }

  // GET /health/specialists
  if (seg0 === "health" && seg1 === "specialists") {
    return NextResponse.json({
      status: "ok",
      endpoint: "mock",
      tnm: true,
      biomarker: true,
      treatment: true,
      models: ["tnm-specialist-lora", "biomarker-specialist-lora", "treatment-specialist-lora"],
    });
  }

  // GET /api/vram
  if (seg0 === "api" && seg1 === "vram" && !seg2) {
    return NextResponse.json(MOCK_VRAM_INFO);
  }

  // GET /api/vram/history
  if (seg0 === "api" && seg1 === "vram" && seg2 === "history") {
    const seconds = parseInt(req.nextUrl.searchParams.get("seconds") ?? "60", 10);
    return NextResponse.json(getMockVramHistory(seconds));
  }

  // GET /memory/cases
  if (seg0 === "memory" && seg1 === "cases") {
    return NextResponse.json({ cases: [] });
  }

  // POST /api/concurrent/run
  if (seg0 === "api" && seg1 === "concurrent" && seg2 === "run" && method === "POST") {
    let cases: string[] = ["lung_adenocarcinoma", "colon_adenocarcinoma", "lung_squamous_cell"];
    try {
      const body = (await req.json()) as { cases?: string[] };
      if (Array.isArray(body.cases)) cases = body.cases;
    } catch { /* use defaults */ }
    const now = Date.now();
    return NextResponse.json({
      run_id: `mock-concurrent-${now}`,
      job_ids: cases.map((c, i) => `mock-${now + i}-${c}`),
      cases,
      started_at: new Date(now).toISOString(),
    });
  }

  // POST /api/counterfactual/{jobId}
  if (seg0 === "api" && seg1 === "counterfactual" && seg2) {
    let hypothesis = "What if the patient were EGFR-positive?";
    try {
      const body = (await req.json()) as { hypothesis?: string };
      if (body.hypothesis) hypothesis = body.hypothesis;
    } catch { /* use default */ }
    return NextResponse.json({
      job_id: seg2,
      hypothesis,
      original_first_line: "PENDING MOLECULAR RESULTS — initiation deferred per NCCN 2024 mandate.",
      revised_plan: {
        treatment_plan: {
          first_line: "Osimertinib 80 mg OD (EGFR exon 19 del confirmed — NCCN Category 1)",
          rationale: "Counterfactual: if EGFR+ confirmed, osimertinib is first-line per FLAURA2.",
          alternatives: ["Osimertinib + Pemetrexed + Carboplatin (FLAURA2 combination arm)"],
        },
      },
      diff_summary:
        "Counterfactual EGFR+ assumption advances treatment from 'pending molecular results' to 'osimertinib first-line'. PFS projection improves from 14 to 18 months (FLAURA2 data).",
    });
  }

  // GET /api/training/reports
  if (seg0 === "api" && seg1 === "training" && seg2 === "reports") {
    return NextResponse.json({
      reports: [
        {
          adapter: "tnm-specialist-lora",
          base_model: "llama3.3:70b-instruct-q4_K_S",
          rank: 16,
          alpha: 32,
          train_loss: 0.142,
          eval_loss: 0.168,
          best_epoch: 3,
          total_steps: 450,
          learning_rate: 2e-4,
          batch_size: 4,
          finished_at: "2026-05-08T14:30:00Z",
        },
        {
          adapter: "biomarker-specialist-lora",
          base_model: "llama3.3:70b-instruct-q4_K_S",
          rank: 16,
          alpha: 32,
          train_loss: 0.138,
          eval_loss: 0.161,
          best_epoch: 4,
          total_steps: 420,
          learning_rate: 2e-4,
          batch_size: 4,
          finished_at: "2026-05-08T16:15:00Z",
        },
        {
          adapter: "treatment-specialist-lora",
          base_model: "llama3.3:70b-instruct-q4_K_S",
          rank: 16,
          alpha: 32,
          train_loss: 0.155,
          eval_loss: 0.182,
          best_epoch: 3,
          total_steps: 480,
          learning_rate: 2e-4,
          batch_size: 4,
          finished_at: "2026-05-08T18:00:00Z",
        },
      ],
    });
  }

  // Fallback for any unrecognised mock path
  return NextResponse.json(
    { detail: `Mock endpoint not implemented: ${method} /${segments.join("/")}` },
    { status: 404 }
  );
}

// ── Live proxy ─────────────────────────────────────────────────────────────────

async function proxy(req: NextRequest, ctx: { params: Promise<{ path?: string[] }> }) {
  const { path: segments } = await ctx.params;
  const segs = segments ?? [];
  const pathPart = segs.join("/");

  // Intercept entirely in mock mode — no upstream call made
  if (process.env.DEMO_MODE === "mock") {
    return handleMock(segs, req);
  }

  const target = `${backendBase()}/${pathPart}${req.nextUrl.search}`;

  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers: filterRequestHeaders(req.headers),
  };

  if (!["GET", "HEAD"].includes(req.method) && req.body) {
    init.body = req.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, { ...init, cache: "no-store" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Upstream fetch failed";
    return NextResponse.json(
      { detail: msg, backend: backendBase(), path: pathPart || "(empty)" },
      { status: 502 }
    );
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: filterResponseHeaders(upstream.headers),
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS",
      "Access-Control-Allow-Headers": "*",
      "Access-Control-Max-Age": "86400",
    },
  });
}
