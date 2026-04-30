"""
hf_space/app.py
===============
Gradio front-end for the Autonomous Oncology Board — HuggingFace Space.

This app wraps the live AOB API (running on AMD MI300X) with a clean
Gradio interface. It connects to the API via HTTP, so the heavy ML
workload stays on the AMD instance; the Space is a pure UI wrapper.

Set the environment variable AOB_API_URL in the HF Space secrets:
    AOB_API_URL=http://<your-amd-host>:8000

If AOB_API_URL is not set, the app will show demo-mode mock responses.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Optional

import gradio as gr

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("AOB_API_URL", "").rstrip("/")
POLL_INTERVAL = 3.0   # seconds between status polls
MAX_WAIT_S    = 600   # 10 minutes max

# ── API helpers ────────────────────────────────────────────────────────────────

def _api_get(path: str) -> Optional[dict]:
    if not _HAS_REQUESTS or not API_URL:
        return None
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _api_post(path: str, payload: dict) -> Optional[dict]:
    if not _HAS_REQUESTS or not API_URL:
        return None
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _check_health() -> str:
    if not API_URL:
        return "⚠️ AOB_API_URL not set — running in demo/mock mode"
    health = _api_get("/health")
    if health:
        ollama = health.get("ollama", "unknown")
        board  = health.get("board_ready", False)
        return f"✅ API connected | Ollama: {ollama} | Board ready: {board}"
    return "❌ Cannot connect to AOB API"


def _get_demo_cases() -> list[str]:
    data = _api_get("/demo/cases")
    if data and "cases" in data:
        return [c["case_name"] for c in data["cases"]]
    return ["lung_adenocarcinoma", "colon_adenocarcinoma", "lung_squamous_cell"]


# ── Mock fallback ──────────────────────────────────────────────────────────────

def _mock_report(case_name: str) -> str:
    """Return a mock management plan for demo/offline mode."""
    mock = {
        "case_id": f"demo_{case_name}",
        "management_plan": {
            "diagnosis": {
                "primary": case_name.replace("_", " ").title(),
                "tnm_stage": "Stage III-IV — pending molecular workup",
                "confidence": 0.87,
            },
            "immediate_actions": [
                "Order comprehensive molecular panel (EGFR/ALK/ROS1/KRAS/PD-L1)",
                "Obtain complete staging CT chest/abdomen/pelvis",
                "Multidisciplinary tumour board review",
            ],
            "treatment_plan": {
                "first_line": "Pending molecular results — osimertinib if EGFR+ / pembrolizumab if PD-L1 ≥50%",
                "rationale": "NCCN Category 1 evidence supports targeted therapy over chemotherapy in driver-mutation positive disease",
                "alternatives": ["Carboplatin + Pemetrexed + Pembrolizumab (driver-negative)", "Clinical trial enrollment"],
            },
            "debate_rounds_completed": 1,
            "consensus_score": 82,
            "disclaimer": "AI research tool. NOT for clinical use.",
        }
    }
    return json.dumps(mock, indent=2)


# ── Core functions ─────────────────────────────────────────────────────────────

def run_demo_case(case_name: str) -> tuple[str, str, str]:
    """
    Run a pre-baked demo case through the AOB pipeline.

    Returns: (status_text, log_text, report_json)
    """
    if not API_URL:
        # Offline/mock mode
        report = _mock_report(case_name)
        return (
            "✅ Mock mode — showing pre-baked response (AOB_API_URL not configured)",
            "🔬 Pathologist → 📚 Researcher → 👨‍⚕️ Oncologist → 🗣️ Debate → ✅ Done",
            report,
        )

    # Submit demo case
    resp = _api_post(f"/demo/run/{case_name}", {})
    if not resp:
        return "❌ Failed to submit demo case", "", ""

    job_id = resp.get("job_id", "")
    if not job_id:
        return "❌ No job_id in response", "", ""

    # Poll for completion
    t0 = time.time()
    log_lines: list[str] = [f"Job {job_id} submitted..."]

    while time.time() - t0 < MAX_WAIT_S:
        time.sleep(POLL_INTERVAL)

        # Get new steps
        status_data = _api_get(f"/status/{job_id}")
        if status_data:
            steps = status_data.get("steps", [])
            if len(steps) > len(log_lines) - 1:
                for step in steps[len(log_lines) - 1:]:
                    agent = step.get("agent", "system")
                    msg   = step.get("message", "")
                    prog  = step.get("progress", 0)
                    log_lines.append(f"[{agent}] ({prog}%) {msg}")

            job_status = status_data.get("status", "")
            if job_status == "done":
                break
            if job_status == "failed":
                err = status_data.get("error", "Unknown error")
                return f"❌ Pipeline failed: {err}", "\n".join(log_lines), ""

    # Fetch final report
    report_data = _api_get(f"/report/{job_id}")
    if not report_data:
        return "❌ Could not fetch report", "\n".join(log_lines), ""

    status_text = (
        f"✅ Complete — {report_data.get('debate_rounds_completed', 0)} debate round(s) | "
        f"Time: {report_data.get('total_time_s', '?')}s"
    )
    return status_text, "\n".join(log_lines), json.dumps(report_data, indent=2)


def run_custom_case(images: list, patient_age: int, sex: str, clinical_notes: str) -> tuple[str, str, str]:
    """
    Run the pipeline on uploaded image patches.

    Returns: (status_text, log_text, report_json)
    """
    if not images:
        return "❌ Please upload at least one image patch", "", ""

    if not API_URL:
        return (
            "⚠️ AOB_API_URL not configured — cannot process custom images in demo mode",
            "",
            "",
        )

    # Encode images to base64
    patches_b64: list[str] = []
    for img in images:
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            patches_b64.append(base64.b64encode(buf.getvalue()).decode())
        except Exception as e:
            return f"❌ Failed to encode image: {e}", "", ""

    payload = {
        "patches_b64": patches_b64,
        "metadata": {
            "patient_age": patient_age,
            "sex": sex,
            "clinical_notes": clinical_notes,
        },
    }

    resp = _api_post("/analyze", payload)
    if not resp:
        return "❌ Failed to submit case to API", "", ""

    job_id = resp.get("job_id", "")

    # Reuse same polling logic
    t0 = time.time()
    log_lines: list[str] = [f"Job {job_id} submitted ({len(patches_b64)} patches)..."]

    while time.time() - t0 < MAX_WAIT_S:
        time.sleep(POLL_INTERVAL)
        status_data = _api_get(f"/status/{job_id}")
        if status_data:
            steps = status_data.get("steps", [])
            for step in steps[len(log_lines) - 1:]:
                agent = step.get("agent", "system")
                msg   = step.get("message", "")
                prog  = step.get("progress", 0)
                log_lines.append(f"[{agent}] ({prog}%) {msg}")

            job_status = status_data.get("status", "")
            if job_status == "done":
                break
            if job_status == "failed":
                err = status_data.get("error", "Unknown error")
                return f"❌ Pipeline failed: {err}", "\n".join(log_lines), ""

    report_data = _api_get(f"/report/{job_id}")
    if not report_data:
        return "❌ Could not fetch report", "\n".join(log_lines), ""

    status_text = (
        f"✅ Complete — {report_data.get('debate_rounds_completed', 0)} debate round(s) | "
        f"Time: {report_data.get('total_time_s', '?')}s"
    )
    return status_text, "\n".join(log_lines), json.dumps(report_data, indent=2)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    demo_cases = _get_demo_cases()
    health_str = _check_health()

    with gr.Blocks(
        title="Autonomous Oncology Board",
        theme=gr.themes.Default(primary_hue="indigo", secondary_hue="purple"),
        css="""
            .header-box { background: linear-gradient(135deg, #4f46e5, #7c3aed);
                          border-radius: 12px; padding: 24px; color: white; margin-bottom: 16px; }
            .disclaimer { background: #fef2f2; border-left: 4px solid #ef4444;
                          padding: 12px; border-radius: 4px; font-size: 0.85em; }
        """,
    ) as app:

        gr.HTML("""
        <div class="header-box">
            <h1 style="margin:0; font-size:1.8em;">🔬 Autonomous Oncology Board</h1>
            <p style="margin:4px 0 0 0; opacity:0.9;">
                3-agent AI tumour board · GigaPath + Llama 3.3 70B · AMD MI300X (192 GB HBM3)
            </p>
        </div>
        """)

        gr.Markdown(f"**API Status:** {health_str}")

        with gr.Tabs():
            # ── Tab 1: Demo Cases ───────────────────────────────────────────
            with gr.TabItem("🎯 Demo Cases"):
                gr.Markdown("""
                Run a pre-baked demo case instantly — no image upload required.
                The pipeline runs GigaPath + Llama 3.3 70B + multi-round debate end-to-end.
                """)

                with gr.Row():
                    demo_dropdown = gr.Dropdown(
                        choices=demo_cases,
                        value=demo_cases[0] if demo_cases else None,
                        label="Select Demo Case",
                    )
                    run_demo_btn = gr.Button("▶️ Run Demo Case", variant="primary")

                demo_status = gr.Textbox(label="Status", interactive=False)
                demo_log    = gr.Textbox(label="Agent Pipeline Log", lines=15, interactive=False)
                demo_report = gr.JSON(label="Management Plan (JSON)")

                run_demo_btn.click(
                    fn=run_demo_case,
                    inputs=[demo_dropdown],
                    outputs=[demo_status, demo_log, demo_report],
                )

            # ── Tab 2: Upload Custom Case ───────────────────────────────────
            with gr.TabItem("📤 Upload Patches"):
                gr.Markdown("""
                Upload your own H&E histopathology patches (224×224 recommended).
                Requires `AOB_API_URL` to be set in Space secrets.
                """)

                with gr.Row():
                    with gr.Column():
                        img_upload  = gr.Gallery(label="Upload Patches (224×224 H&E)", type="pil")
                        age_input   = gr.Slider(18, 100, value=65, label="Patient Age", step=1)
                        sex_input   = gr.Radio(["M", "F", "Other"], value="M", label="Sex")
                        notes_input = gr.Textbox(label="Clinical Notes", placeholder="e.g. Peripheral lung mass, non-smoker, EGFR pending...")
                        run_btn     = gr.Button("▶️ Analyse Case", variant="primary")

                    with gr.Column():
                        upload_status = gr.Textbox(label="Status", interactive=False)
                        upload_log    = gr.Textbox(label="Agent Pipeline Log", lines=12, interactive=False)

                upload_report = gr.JSON(label="Management Plan (JSON)")

                run_btn.click(
                    fn=run_custom_case,
                    inputs=[img_upload, age_input, sex_input, notes_input],
                    outputs=[upload_status, upload_log, upload_report],
                )

            # ── Tab 3: About ────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## How It Works

                The Autonomous Oncology Board (AOB) simulates a multi-disciplinary tumour board meeting:

                1. **Agent 1 — Pathologist** (Prov-GigaPath ViT-Giant)
                   - Analyses histopathology image patches
                   - Returns tissue type, confidence, biomarker scores, attention heatmaps
                   - Runs MC Dropout for uncertainty quantification

                2. **Agent 2 — Researcher** (RAG + Llama 3.3 70B)
                   - Retrieves NCCN guideline evidence via Qdrant vector search
                   - Synthesises treatment options, evidence quality, citations

                3. **Agent 3 — Oncologist** (Llama 3.3 70B)
                   - Synthesises agents 1+2 into a complete Patient Management Plan
                   - Incorporates similar past cases from Board Memory

                4. **Agent Debate Loop** (up to 3 rounds)
                   - Researcher challenges the plan against NCCN guidelines
                   - Pathologist referee re-evaluates morphological findings
                   - Oncologist revises the plan
                   - MetaEvaluator scores consensus (0–100, threshold 70)

                ## Hardware

                Runs on **AMD Instinct MI300X** with 192 GB HBM3 unified memory.
                The only GPU that can hold GigaPath + Llama 70B simultaneously.

                | Model | VRAM |
                |---|---|
                | Prov-GigaPath ViT-Giant | ~3 GB |
                | Llama 3.3 70B (Ollama) | ~40 GB |
                | **H100 VRAM limit** | **80 GB ❌** |

                ## Links
                - 📦 [GitHub Repository](https://github.com/Aurnobb-Tasneem/autonomous-oncology-board)
                - 🏆 AMD MI300X Hackathon 2026

                ---

                <div class="disclaimer">
                ⚠️ <strong>DISCLAIMER:</strong> This is an AI research demonstration. NOT a medical device.
                NOT approved for clinical use. All outputs must be reviewed by a qualified oncologist.
                </div>
                """)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
