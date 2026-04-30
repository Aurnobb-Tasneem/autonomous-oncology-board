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
    yield
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
def _run_board_job(job: Job, images_bytes: list[bytes]):
    """Runs the AOB pipeline in a background thread, emitting steps to the job."""
    from PIL import Image

    try:
        job.status = JobStatus.RUNNING
        job.add_step("system", "Case received — starting analysis pipeline", 5)

        # ── Load images ──────────────────────────────────────────────────────
        job.add_step("system", f"Decoding {len(images_bytes)} image patches", 8)
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]

        # ── Monkey-patch board to emit steps ─────────────────────────────────
        # We intercept logging to emit SSE steps
        original_run = _board.run

        def instrumented_run(case_id, imgs, batch_size=16):
            job.add_step("pathologist", "GigaPath: loading model and preprocessing patches", 15)
            # Run pathologist
            path_report = _board.pathologist.analyse(case_id, imgs, batch_size=batch_size)
            job.add_step(
                "pathologist",
                f"GigaPath: classified {path_report.n_patches} patches → "
                f"{path_report.tissue_type.replace('_', ' ').title()} "
                f"({path_report.confidence:.0%} confidence)",
                40,
            )
            if path_report.flags:
                job.add_step("pathologist", f"⚠️ Flags detected: {', '.join(path_report.flags)}", 42)

            job.add_step("researcher", "Building clinical query from pathology findings", 45)
            research = _board.researcher.research(path_report)
            job.add_step(
                "researcher",
                f"Retrieved {research.raw_evidence.get('n_retrieved', 0)} evidence documents "
                f"(quality: {research.evidence_quality})",
                60,
            )
            job.add_step("researcher", f"Synthesised {len(research.treatment_options)} treatment options", 65)

            job.add_step("oncologist", "Llama 3.3 70B: generating Patient Management Plan...", 70)
            plan = _board.oncologist.synthesise(path_report, research)
            job.add_step(
                "oncologist",
                f"Management plan complete — confidence: {plan.confidence_score:.0%}",
                95,
            )

            from ml.board import BoardResult
            import time
            return BoardResult(
                case_id=case_id,
                pathology_report=path_report,
                research_summary=research,
                management_plan=plan,
                total_time_s=0,  # board.run will set this
            )

        result = instrumented_run(job.case_id, images)
        job.result = result
        job.status = JobStatus.DONE
        job.add_step("system", f"✅ Analysis complete — {result.management_plan.diagnosis.primary}", 100)

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
        args=(job, images_bytes),
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
