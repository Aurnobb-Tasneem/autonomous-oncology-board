"""
ml/api.py
=========
FastAPI REST layer for the Autonomous Oncology Board.

Endpoints:
  POST /analyze              — Submit a case (image patches + metadata)
  GET  /status/{job_id}      — SSE stream of live agent step updates
  GET  /report/{job_id}      — Retrieve completed ManagementPlan JSON
  GET  /health               — Health check (Ollama + GigaPath status)
  GET  /cases                — List all completed cases

Usage (run inside Docker container on port 8000):
  cd /workspace/aob
  export PYTHONPATH=/workspace/aob
  export OLLAMA_HOST=http://172.17.0.1:11434
  export HF_TOKEN=hf_...
  uvicorn ml.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from ml.board import AutonomousOncologyBoard, BoardResult
from ml.models.llm_client import OllamaClient
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("aob_api")


# ── Job state machine ────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"


class AgentStep(BaseModel):
    agent: str          # "pathologist" | "researcher" | "oncologist" | "system"
    message: str
    timestamp: str
    progress: int       # 0–100


class Job:
    def __init__(self, job_id: str, case_id: str):
        self.job_id    = job_id
        self.case_id   = case_id
        self.status    = JobStatus.QUEUED
        self.steps: list[AgentStep] = []
        self.result: Optional[BoardResult] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._lock = threading.Lock()
        # Cached on-demand by GET /attention_scores/{job_id}
        self.attention_scores_14x14: list = []

    def add_step(self, agent: str, message: str, progress: int):
        step = AgentStep(
            agent=agent,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            progress=progress,
        )
        with self._lock:
            self.steps.append(step)
        log.info(f"[{self.job_id}] [{agent}] {message}")
        return step


# ── In-memory job store (replace with Redis for production) ──────────────────
_jobs: dict[str, Job] = {}
_board: Optional[AutonomousOncologyBoard] = None

# ── VRAM timeseries ring buffer (polled every 2s by background thread) ────────
_VRAM_RING_SIZE = 300   # 300 × 2s = 10 minutes rolling window
_vram_history: deque = deque(maxlen=_VRAM_RING_SIZE)
_vram_monitor_running = False


def _vram_monitor_loop():
    """Background thread: poll rocm-smi every 2s and append to ring buffer."""
    import subprocess
    import re
    global _vram_monitor_running
    _vram_monitor_running = True
    while _vram_monitor_running:
        ts = time.time()
        vram_used_gb  = 0.0
        vram_total_gb = 192.0
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for card in data.values():
                    used_bytes  = int(card.get("VRAM Total Used Memory (B)", 0))
                    total_bytes = int(card.get("VRAM Total Memory (B)", 192 * 1024**3))
                    vram_used_gb  = round(used_bytes  / (1024 ** 3), 2)
                    vram_total_gb = round(total_bytes / (1024 ** 3), 2)
                    break
        except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired, Exception):
            # Fallback: try nvidia-smi or just report 0
            try:
                import torch
                if torch.cuda.is_available():
                    vram_used_gb  = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
                    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            except Exception:
                pass

        _vram_history.append({
            "ts": ts,
            "used_gb":  vram_used_gb,
            "total_gb": vram_total_gb,
            "pct":      round(vram_used_gb / vram_total_gb * 100, 1) if vram_total_gb > 0 else 0.0,
        })
        time.sleep(2)


# ── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _board
    log.info("AOB API: starting up...")
    _board = AutonomousOncologyBoard(
        hf_token=os.getenv("HF_TOKEN", ""),
        ollama_host=os.getenv("OLLAMA_HOST", "http://172.17.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.3:70b"),
    )
    log.info("AOB API: board initialised — ready to accept cases")

    # Start VRAM monitor background thread
    vram_thread = threading.Thread(target=_vram_monitor_loop, daemon=True)
    vram_thread.start()
    log.info("AOB API: VRAM monitor started (polling every 2s)")

    yield

    global _vram_monitor_running
    _vram_monitor_running = False
    log.info("AOB API: shutting down")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Oncology Board API",
    description="Multi-agent AI tumour board running GigaPath + Llama 3.3 70B on AMD MI300X (ROCm)",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the demo frontend at /
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/")
async def serve_demo():
    return FileResponse(str(_static_dir / "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    case_id: Optional[str] = Field(
        default=None,
        description="Optional case identifier. Auto-generated if not provided."
    )
    # Base64-encoded JPEG/PNG patch images
    patches_b64: list[str] = Field(
        description="List of base64-encoded image patches (224×224 recommended).",
        min_length=1,
        max_length=64,
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional clinical metadata (age, sex, clinical_notes, etc.)",
    )


class AnalyzeResponse(BaseModel):
    job_id: str
    case_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    case_id: str
    status: JobStatus
    steps: list[AgentStep]
    created_at: str
    error: Optional[str] = None


# ── Background worker ────────────────────────────────────────────────────────
def _run_board_job(job: Job, images_bytes: list[bytes], metadata: dict):
    """Runs the AOB pipeline in a background thread, emitting steps via SSE."""
    from PIL import Image

    try:
        job.status = JobStatus.RUNNING
        job.add_step("system", "Case received — starting AOB pipeline", 5)

        # Decode images
        job.add_step("system", f"Decoding {len(images_bytes)} image patches", 8)
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]

        # Step callback — bridges board.run() steps into the SSE job stream
        def step_cb(agent: str, message: str, progress: int):
            job.add_step(agent, message, progress)

        # Run the full pipeline + Agent Debate loop
        result = _board.run(
            case_id=job.case_id,
            images=images,
            debate_mode=True,
            step_callback=step_cb,
            metadata=metadata,
        )

        job.result = result
        job.status = JobStatus.DONE

    except Exception as e:
        log.exception(f"Job {job.job_id} failed: {e}")
        job.status = JobStatus.FAILED
        job.error  = str(e)
        job.add_step("system", f"❌ Pipeline failed: {e}", -1)



# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check API, Ollama, and GigaPath status."""
    llm_ok = OllamaClient().ping()
    return {
        "status": "ok",
        "ollama": "connected" if llm_ok else "unreachable",
        "model": os.getenv("OLLAMA_MODEL", "llama3.3:70b"),
        "board_ready": _board is not None,
        "active_jobs": len([j for j in _jobs.values() if j.status == JobStatus.RUNNING]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Submit a case for analysis.

    Accepts a list of base64-encoded image patches and optional metadata.
    Returns a job_id to poll for status and results.

    Example:
        import base64, requests
        with open("patch.jpg", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = requests.post("http://localhost:8000/analyze", json={
            "patches_b64": [b64],
            "metadata": {"patient_age": 67, "sex": "M"}
        })
        job_id = r.json()["job_id"]
    """
    if _board is None:
        raise HTTPException(status_code=503, detail="Board not initialised yet")

    case_id = request.case_id or f"case_{uuid.uuid4().hex[:8]}"
    job_id  = f"job_{uuid.uuid4().hex[:12]}"

    # Decode base64 patches
    try:
        images_bytes = [base64.b64decode(p) for p in request.patches_b64]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Create job
    job = Job(job_id=job_id, case_id=case_id)
    _jobs[job_id] = job

    # Run in background thread (not async — GigaPath/Ollama are sync)
    thread = threading.Thread(
        target=_run_board_job,
        args=(job, images_bytes, request.metadata),
        daemon=True,
        name=f"board-{job_id}",
    )
    thread.start()

    return AnalyzeResponse(
        job_id=job_id,
        case_id=case_id,
        status=JobStatus.QUEUED,
        message=f"Case queued. Poll /status/{job_id} or stream /stream/{job_id}",
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Poll for job status and completed steps."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(
        job_id=job.job_id,
        case_id=job.case_id,
        status=job.status,
        steps=job.steps,
        created_at=job.created_at,
        error=job.error,
    )


@app.get("/stream/{job_id}")
async def stream_status(job_id: str):
    """
    SSE stream of real-time agent step updates.

    Connect with EventSource in the browser:
        const es = new EventSource(`/stream/${jobId}`);
        es.onmessage = e => console.log(JSON.parse(e.data));
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        last_idx = 0
        while True:
            with job._lock:
                new_steps = job.steps[last_idx:]
                current_status = job.status

            for step in new_steps:
                data = json.dumps(step.model_dump())
                yield f"data: {data}\n\n"
                last_idx += 1

            if current_status in (JobStatus.DONE, JobStatus.FAILED):
                # Send final status event
                yield f"event: done\ndata: {json.dumps({'status': current_status, 'job_id': job_id})}\n\n"
                break

            await __import__("asyncio").sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    """
    Retrieve the completed ManagementPlan JSON.
    Returns 202 if still processing, 200 when done, 500 if failed.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status == JobStatus.RUNNING or job.status == JobStatus.QUEUED:
        return JSONResponse(
            status_code=202,
            content={"status": job.status, "message": "Still processing"},
        )
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error)

    return job.result.to_dict()


@app.get("/cases")
async def list_cases():
    """List all jobs with their status."""
    return [
        {
            "job_id": j.job_id,
            "case_id": j.case_id,
            "status": j.status,
            "created_at": j.created_at,
            "n_steps": len(j.steps),
        }
        for j in _jobs.values()
    ]


@app.get("/memory/cases")
async def list_memory_cases():
    """
    List all cases stored in Board Memory.

    Returns metadata for every case the board has processed, sorted
    newest-first. Centroid vectors are excluded from the response.
    Useful for auditing the board's institutional memory.
    """
    if _board is None:
        raise HTTPException(status_code=503, detail="Board not initialised yet")
    cases = _board.memory.list_all()
    return {
        "total": len(cases),
        "cases": cases,
    }


@app.get("/heatmaps/{job_id}")
async def get_heatmaps(job_id: str):
    """
    Retrieve GigaPath attention heatmaps for a completed job.

    Returns a list of base64-encoded PNG images — one per patch.
    High-attention regions are highlighted in red and labeled '⚠ SUSPICIOUS'.

    Returns 202 if still processing, 404 if not found.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        return JSONResponse(status_code=202, content={"status": job.status, "message": "Still processing"})

    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error)

    heatmaps = job.result.heatmaps_b64 if job.result else []
    return {
        "job_id": job_id,
        "case_id": job.case_id,
        "n_heatmaps": len(heatmaps),
        "heatmaps_b64": heatmaps,
        "description": "Base64-encoded PNG overlays. Decode and display as <img src='data:image/png;base64,...'>",
    }


@app.get("/api/vram")
async def get_vram():
    """
    Live VRAM usage from rocm-smi.

    Returns current VRAM consumption by all models on the AMD MI300X.
    Includes H100 comparison limit (80 GB) for the demo.

    Poll this endpoint every 2 seconds to animate the VRAM dashboard.
    """
    import subprocess

    import re as _re

    # ── Total VRAM (JSON — reliable on all ROCm versions) ─────────────────
    try:
        mem_result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if mem_result.returncode != 0:
            raise RuntimeError(mem_result.stderr)
        import json as _json
        raw = _json.loads(mem_result.stdout)
        card     = next(iter(raw.values()), {})
        total_b  = int(card.get("VRAM Total Memory (B)", 0))
        used_b   = int(card.get("VRAM Total Used Memory (B)", 0))
        total_gb = round(total_b / 1e9, 1)
        used_gb  = round(used_b  / 1e9, 1)
        pct      = round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0
        source   = "rocm-smi"
    except Exception as e:
        log.warning(f"rocm-smi --showmeminfo failed: {e} — returning mock data")
        used_gb  = 102.3
        total_gb = 191.7
        pct      = 53.4
        source   = "mock"

    # ── Per-process VRAM (rocm-smi --showpids text output) ────────────────
    # Format (header + data rows):
    #   PID     PROCESS NAME    GPU(s)  VRAM USED       SDMA USED  CU OCCUPANCY
    #   14174   uvicorn         1       20796682240     0           0
    #   14405   ollama          1       48175480832     0           0
    processes: dict[str, float] = {}   # {process_name: vram_gb}
    try:
        pid_result = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True, text=True, timeout=5,
        )
        for line in pid_result.stdout.splitlines():
            # Match lines that start with a PID (integer), followed by process name and VRAM bytes
            m = _re.match(r"^\s*(\d+)\s+(\S+)\s+\d+\s+(\d+)", line)
            if m:
                proc_name = m.group(2).lower()
                vram_b    = int(m.group(3))
                vram_gb   = round(vram_b / 1e9, 1)
                processes[proc_name] = processes.get(proc_name, 0.0) + vram_gb
    except Exception as e:
        log.debug(f"rocm-smi --showpids unavailable: {e}")

    # ── Derive per-model breakdown from real process data ─────────────────
    # GigaPath lives in the uvicorn process; Llama 70B lives in ollama.
    # KV cache = ollama total − Llama weights; runtime overhead = uvicorn − GigaPath.
    GIGAPATH_WEIGHTS_GB = 3.2
    LLAMA_70B_WEIGHTS_GB = 40.0

    uvicorn_gb = processes.get("uvicorn", 0.0) or processes.get("python", 0.0) or processes.get("python3", 0.0)
    ollama_gb  = processes.get("ollama", 0.0)

    if ollama_gb > 0:
        kv_cache_gb      = max(0.0, round(ollama_gb  - LLAMA_70B_WEIGHTS_GB, 1))
        runtime_overhead = max(0.0, round(uvicorn_gb - GIGAPATH_WEIGHTS_GB, 1))
    else:
        # Fall back to formula when process data is unavailable
        kv_cache_gb      = max(0.0, round(used_gb - (GIGAPATH_WEIGHTS_GB + LLAMA_70B_WEIGHTS_GB), 1))
        runtime_overhead = None

    return {
        "used_gb":      used_gb,
        "total_gb":     total_gb,
        "free_gb":      round(total_gb - used_gb, 1),
        "percent":      pct,
        "percent_used": pct,
        "source":       source,
        "h100_limit_gb":  80.0,
        "exceeds_h100":   used_gb > 80.0,
        "model_breakdown": {
            "gigapath_gb":        GIGAPATH_WEIGHTS_GB,
            "llama_gb":           LLAMA_70B_WEIGHTS_GB,
            "kv_cache_gb":        kv_cache_gb,
            "runtime_overhead_gb": runtime_overhead,
        },
        # Actual per-process breakdown from rocm-smi --showpids
        "processes": {k: v for k, v in processes.items()} if processes else None,
        "hardware": "AMD Instinct MI300X · 192 GB HBM3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



# ── Demo endpoints ─────────────────────────────────────────────────────────────
_DEMO_CASES_DIR = Path(__file__).parent.parent / "data" / "demo_cases"


@app.get("/demo/cases")
async def list_demo_cases():
    """
    List all available pre-baked demo cases.

    Returns case metadata without the full patch data.
    Use POST /demo/run/{case_name} to run a demo case through the full pipeline.
    """
    if not _DEMO_CASES_DIR.exists():
        return {"cases": [], "message": "No demo cases directory found"}

    cases = []
    for json_file in sorted(_DEMO_CASES_DIR.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            cases.append({
                "case_name":    data.get("case_name", json_file.stem),
                "display_name": data.get("display_name", json_file.stem),
                "tissue_type":  data.get("tissue_type", "unknown"),
                "description":  data.get("description", ""),
                "n_patches":    len(data.get("patches_b64", [])),
                "metadata":     data.get("metadata", {}),
            })
        except Exception as e:
            log.warning(f"Could not load demo case {json_file}: {e}")

    return {"total": len(cases), "cases": cases}


@app.post("/demo/run/{case_name}", response_model=AnalyzeResponse)
async def run_demo_case(case_name: str, background_tasks: BackgroundTasks):
    """
    Run the full AOB pipeline on a pre-baked demo case.

    No image upload needed — patches are loaded from data/demo_cases/{case_name}.json.
    Returns a job_id immediately; poll /stream/{job_id} or /report/{job_id}.

    Available case names: lung_adenocarcinoma, colon_adenocarcinoma, lung_squamous_cell
    """
    if _board is None:
        raise HTTPException(status_code=503, detail="Board not initialised yet")

    case_file = _DEMO_CASES_DIR / f"{case_name}.json"
    if not case_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Demo case '{case_name}' not found. "
                   f"Available: {[f.stem for f in _DEMO_CASES_DIR.glob('*.json')]}"
        )

    try:
        with open(case_file, encoding="utf-8") as f:
            demo_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load demo case: {e}")

    patches_b64: list[str] = demo_data.get("patches_b64", [])
    if not patches_b64:
        raise HTTPException(status_code=500, detail="Demo case has no patches")

    # Decode patches
    try:
        images_bytes = [base64.b64decode(p) for p in patches_b64]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode demo patches: {e}")

    # Create and register job
    case_id = f"demo_{case_name}_{uuid.uuid4().hex[:6]}"
    job_id  = f"job_{uuid.uuid4().hex[:12]}"
    job = Job(job_id=job_id, case_id=case_id)
    _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_board_job,
        args=(job, images_bytes, demo_data.get("metadata", {})),
        daemon=True,
        name=f"demo-{job_id}",
    )
    thread.start()

    return AnalyzeResponse(
        job_id=job_id,
        case_id=case_id,
        status=JobStatus.QUEUED,
        message=f"Demo case '{case_name}' queued. Stream: /stream/{job_id}",
    )


# ── Attention Scores Endpoint (Task 3) ──────────────────────────────────────

@app.get(
    "/attention_scores/{job_id}",
    summary="Raw last-block attention scores for frontend saliency rendering",
    tags=["saliency"],
)
async def get_attention_scores(job_id: str):
    """
    Return raw GigaPath last-block CLS attention scores for a completed job.

    These are **not** the pre-rendered heatmap PNGs stored in the job result —
    those are full Attention Rollout across all ViT blocks, returned as base64
    PNG overlays.

    This endpoint returns the raw 14×14 float grid from `blocks[-1].attn` only,
    which is what frontend saliency libraries (DINO-style, WebGL canvas) need for
    interactive, client-side rendering.

    The scores are computed **on demand** when this endpoint is first called for a
    given job, then cached in the Job object to avoid re-running the forward pass.

    Args:
        job_id: The job ID returned by POST /analyze or POST /analyze/demo.

    Returns:
        JSON with:
          job_id      (str)
          case_id     (str)
          n_patches   (int)  — number of patches scored
          grid_size   (int)  — always 14 for GigaPath (ViT-Giant, patch_size=16)
          scores      (list) — N × 14 × 14 floats in [0, 1]
          description (str)  — human-readable usage hint

    Raises:
        404 if job not found.
        409 if job is not yet complete.
        500 on scoring failure.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job '{job_id}' is not complete yet (status={job.status}). "
                "Wait until status=done before requesting attention scores."
            ),
        )

    # Return cached result if already computed
    if job.attention_scores_14x14:
        return {
            "job_id":      job_id,
            "case_id":     job.case_id,
            "n_patches":   len(job.attention_scores_14x14),
            "grid_size":   14,
            "scores":      job.attention_scores_14x14,
            "cached":      True,
            "description": (
                "Raw last-block CLS attention scores (14×14 per patch). "
                "scores[patch][row][col] ∈ [0, 1]. "
                "Use for frontend WebGL/canvas saliency rendering."
            ),
        }

    # Compute on demand — we need the original images from the job result.
    # The board stores heatmap PNGs; we reconstruct PIL images from them to
    # avoid re-uploading. Fallback: re-run on stored result patches if available.
    if job.result is None:
        raise HTTPException(
            status_code=500,
            detail="Job completed but result is missing — cannot compute scores.",
        )

    try:
        from PIL import Image
        import io
        import base64

        # Reconstruct PIL images from the stored base64 heatmap PNGs.
        # We use these as proxies for the original patch images since they
        # are the same spatial extent (224×224).  For real production use,
        # the original patch bytes should be stored separately.
        heatmaps_b64: list[str] = job.result.heatmaps_b64 or []

        if not heatmaps_b64:
            raise ValueError("No heatmaps stored in job result — cannot derive patches")

        images: list[Image.Image] = []
        for b64_str in heatmaps_b64:
            raw = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(img)

        if _board is None:
            raise RuntimeError("Board not initialised")

        scores = _board.pathologist.generate_raw_attention_scores(images)

        # Cache so subsequent calls are instant
        with job._lock:
            job.attention_scores_14x14 = scores

        return {
            "job_id":      job_id,
            "case_id":     job.case_id,
            "n_patches":   len(scores),
            "grid_size":   14,
            "scores":      scores,
            "cached":      False,
            "description": (
                "Raw last-block CLS attention scores (14×14 per patch). "
                "scores[patch][row][col] ∈ [0, 1]. "
                "Use for frontend WebGL/canvas saliency rendering."
            ),
        }

    except Exception as e:
        log.exception(f"GET /attention_scores/{job_id}: scoring failed — {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Attention score extraction failed: {e}",
        )


# ── VRAM Timeseries ───────────────────────────────────────────────────────────

@app.get("/api/vram/history")
async def vram_history(seconds: int = 300):
    """
    GET /api/vram/history?seconds=300

    Returns the rolling VRAM usage timeseries for the last N seconds.

    Response:
        {
          "points": [{"ts": float, "used_gb": float, "total_gb": float, "pct": float}],
          "current_gb": float,
          "total_gb": float,
          "h100_limit_gb": 80,
          "mi300x_total_gb": 192
        }
    """
    cutoff = time.time() - seconds
    points = [p for p in list(_vram_history) if p["ts"] >= cutoff]

    current_gb = points[-1]["used_gb"] if points else 0.0
    total_gb   = points[-1]["total_gb"] if points else 192.0

    return JSONResponse({
        "points":          points,
        "current_gb":      current_gb,
        "total_gb":        total_gb,
        "h100_limit_gb":   80,       # The comparison line for the UI
        "mi300x_total_gb": 192,
        "oom_if_h100":     current_gb > 80,
    })


@app.get("/api/vram/current")
async def vram_current():
    """GET /api/vram/current — single VRAM reading."""
    if _vram_history:
        latest = _vram_history[-1]
    else:
        latest = {"ts": time.time(), "used_gb": 0.0, "total_gb": 192.0, "pct": 0.0}
    return JSONResponse({**latest, "h100_limit_gb": 80, "oom_if_h100": latest["used_gb"] > 80})


# ── Concurrent Multi-Case Stress Test ─────────────────────────────────────────

class ConcurrentRunRequest(BaseModel):
    cases: list[dict] = Field(
        description="List of case configs [{case_id, patches_b64, metadata}]. Default: 5 demo cases.",
        default=None,
    )
    n_demo_cases: int = Field(
        default=5,
        description="If cases is None, generate this many demo cases.",
        ge=1, le=10,
    )


@app.post("/api/concurrent/run")
async def concurrent_run(
    request: ConcurrentRunRequest,
    background_tasks: BackgroundTasks,
):
    """
    POST /api/concurrent/run

    Fire multiple AOB analysis jobs simultaneously and return a bundle with:
    - Per-case job IDs (poll /status/{job_id} for streaming)
    - Peak VRAM usage during the run
    - Wall-clock timing per case

    This endpoint is the live proof that multiple cases can run concurrently
    on the MI300X 192 GB unified memory.

    Response:
        {
          "run_id": str,
          "n_cases": int,
          "job_ids": [str],
          "vram_at_start_gb": float,
          "message": str
        }
    """
    if _board is None:
        raise HTTPException(status_code=503, detail="Board not initialised")

    run_id = f"concurrent_{uuid.uuid4().hex[:8]}"

    # Build case list
    cases = request.cases
    if cases is None:
        cases = _build_demo_cases(request.n_demo_cases)

    vram_at_start = _vram_history[-1]["used_gb"] if _vram_history else 0.0

    job_ids = []
    for case_cfg in cases:
        case_id     = case_cfg.get("case_id") or f"{run_id}_{len(job_ids)}"
        patches_b64 = case_cfg.get("patches_b64") or _make_demo_patches(4)
        metadata    = case_cfg.get("metadata") or {}

        # Decode patches
        try:
            images_bytes = [base64.b64decode(p) for p in patches_b64]
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Bad base64 in case {case_id}: {e}")

        job_id = uuid.uuid4().hex[:12]
        job    = Job(job_id, case_id)
        _jobs[job_id] = job

        threading.Thread(
            target=_run_board_job,
            args=(job, images_bytes, metadata),
            daemon=True,
        ).start()

        job_ids.append(job_id)

    return JSONResponse({
        "run_id":           run_id,
        "n_cases":          len(job_ids),
        "job_ids":          job_ids,
        "vram_at_start_gb": vram_at_start,
        "message": (
            f"{len(job_ids)} cases launched simultaneously. "
            "Poll /api/vram/history to watch VRAM climb in real time. "
            "H100 OOM threshold: 80 GB."
        ),
    })


def _build_demo_cases(n: int) -> list[dict]:
    """Generate N demo case configurations for the concurrent stress test."""
    import random
    demo_cases = [
        {"case_id": "DEMO-LUNG-ADENO",    "metadata": {"tissue_type": "lung_adenocarcinoma",           "age": 63, "sex": "M"}},
        {"case_id": "DEMO-COLON-ADENO",   "metadata": {"tissue_type": "colon_adenocarcinoma",          "age": 58, "sex": "F"}},
        {"case_id": "DEMO-LUNG-SCC",      "metadata": {"tissue_type": "lung_squamous_cell_carcinoma",  "age": 71, "sex": "M"}},
        {"case_id": "DEMO-LUNG-BENIGN",   "metadata": {"tissue_type": "lung_benign_tissue",            "age": 45, "sex": "F"}},
        {"case_id": "DEMO-COLON-BENIGN",  "metadata": {"tissue_type": "colon_benign_tissue",           "age": 52, "sex": "M"}},
    ]
    return [
        {**demo_cases[i % len(demo_cases)], "patches_b64": _make_demo_patches(4)}
        for i in range(n)
    ]


def _make_demo_patches(n: int) -> list[str]:
    """Generate N random 224×224 RGB patches as base64 JPEG."""
    import io as _io
    import numpy as _np
    from PIL import Image as _Img

    patches = []
    for _ in range(n):
        arr = _np.random.randint(100, 255, (224, 224, 3), dtype=_np.uint8)
        buf = _io.BytesIO()
        _Img.fromarray(arr).save(buf, format="JPEG", quality=85)
        patches.append(base64.b64encode(buf.getvalue()).decode())
    return patches


# ── Training reports & benchmark endpoints ─────────────────────────────────────

@app.get("/api/training/reports")
async def training_reports():
    """Return all LoRA adapter training reports found in checkpoints dir."""
    checkpoints_dir = Path(__file__).parent / "models" / "checkpoints"
    reports = {}
    for task in ["tnm_lora", "biomarker_lora", "treatment_lora", "giga_head"]:
        report_path = checkpoints_dir / task / "training_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    reports[task] = json.load(f)
            except Exception:
                reports[task] = {"error": "Could not parse report"}
        else:
            reports[task] = {"status": "not_trained_yet"}
    return JSONResponse({"reports": reports})


@app.get("/api/benchmark/latest")
async def benchmark_latest():
    """Return the latest ClinicalEval benchmark results if available."""
    results_dir = Path(__file__).parent.parent / "eval" / "results"
    eval_files  = sorted(results_dir.glob("clinical_eval_*.json")) if results_dir.exists() else []
    if not eval_files:
        return JSONResponse({"status": "no_results_yet", "hint": "Run: python -m eval.clinical_eval"})

    latest = eval_files[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
        return JSONResponse(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse benchmark results: {e}")


@app.post("/api/counterfactual/{job_id}")
async def run_counterfactual(job_id: str, edits: dict):
    """
    POST /api/counterfactual/{job_id}

    Run counterfactual replanning on a completed case.

    Body: {"egfr_status": "EGFR negative", "tnm_stage": "Stage II"}

    Returns the CounterfactualPlan JSON.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = _jobs[job_id]
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail="Job not complete")
    if job.result is None or job.result.management_plan is None:
        raise HTTPException(status_code=500, detail="No management plan found")

    try:
        if _board is None:
            raise RuntimeError("Board not initialised")
        from ml.agents.counterfactual import CounterfactualAgent
        agent  = CounterfactualAgent(llm_client=_board.llm)
        result = agent.replan(job.result.management_plan, edits)
        return JSONResponse(result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WSI analysis endpoint ─────────────────────────────────────────────────────

class WSIAnalyzeRequest(BaseModel):
    slide_path: str = Field(description="Path to .svs/.ndpi/.tiff WSI file on server filesystem")
    case_id:    Optional[str] = None
    metadata:   dict = Field(default_factory=dict)
    max_patches: int = Field(default=64, ge=1, le=512)


@app.post("/analyze/wsi")
async def analyze_wsi(request: WSIAnalyzeRequest, background_tasks: BackgroundTasks):
    """
    POST /analyze/wsi

    Accepts a server-side WSI file path, extracts patches via openslide,
    and runs the full AOB pipeline. Returns a job_id for SSE streaming.
    """
    if _board is None:
        raise HTTPException(status_code=503, detail="Board not initialised")

    slide_path = Path(request.slide_path)
    if not slide_path.exists():
        raise HTTPException(status_code=404, detail=f"Slide not found: {request.slide_path}")

    case_id = request.case_id or f"wsi_{slide_path.stem}"
    job_id  = uuid.uuid4().hex[:12]
    job     = Job(job_id, case_id)
    _jobs[job_id] = job

    def _run_wsi():
        from PIL import Image
        try:
            from ml.data.wsi import extract_patches
        except ImportError:
            job.status = JobStatus.FAILED
            job.error  = "openslide-python not installed"
            return

        images_bytes = []
        try:
            for patch_img, (x, y) in extract_patches(
                slide_path, max_patches=request.max_patches
            ):
                buf = io.BytesIO()
                patch_img.save(buf, format="JPEG")
                images_bytes.append(buf.getvalue())
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error  = str(e)
            return

        if not images_bytes:
            job.status = JobStatus.FAILED
            job.error  = "No tissue patches extracted from slide"
            return

        _run_board_job(job, images_bytes, request.metadata)

    threading.Thread(target=_run_wsi, daemon=True).start()

    return AnalyzeResponse(
        job_id=job_id,
        case_id=case_id,
        status=JobStatus.QUEUED,
        message=f"WSI analysis queued: {slide_path.name} ({request.max_patches} max patches)",
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ml.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
