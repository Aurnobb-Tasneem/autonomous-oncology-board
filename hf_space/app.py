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

def _mock_stages() -> list[dict]:
    """
    Return ordered pipeline steps matching the real board.py emit() sequence.
    Each dict: 'delay_s' (sleep before emitting), 'agent', 'message', 'log_line'.
    Total elapsed: ~35 seconds, matching live MI300X timing.
    Includes all real agents: pathologist, vlm_pathologist, researcher,
    tnm_specialist, biomarker_specialist, differential, treatment_specialist,
    oncologist, patient_summary, trial_matcher, counterfactual, system.
    """
    return [
        {
            "delay_s": 0.5,
            "agent": "system",
            "message": "⏳ Board session initialised — AMD MI300X 192 GB HBM3 unified memory",
            "log_line": "[system] (2%) Board session initialised — AMD MI300X 192 GB HBM3",
        },
        {
            "delay_s": 1.2,
            "agent": "pathologist",
            "message": "🔬 GigaPath: loading model and preprocessing 12 patches",
            "log_line": "[pathologist] (8%) GigaPath: loading model and preprocessing 12 patches",
        },
        {
            "delay_s": 2.3,
            "agent": "pathologist",
            "message": "🔬 GigaPath: 12 patches analysed → Lung Adenocarcinoma (94% confidence)",
            "log_line": "[pathologist] (30%) 12 patches analysed → Lung Adenocarcinoma (94%)",
        },
        {
            "delay_s": 1.0,
            "agent": "pathologist",
            "message": "🔥 9 attention heatmaps generated — suspicious regions highlighted in red",
            "log_line": "[pathologist] (35%) 9 heatmaps generated — suspicious regions in red",
        },
        {
            "delay_s": 1.0,
            "agent": "pathologist",
            "message": "🔬 MC Dropout ×20: Uncertainty 94.2% ± 3.1% (low) — biopsy not required",
            "log_line": "[pathologist] (38%) MC Dropout ×20: 94.2% ± 3.1% (uncertainty: low)",
        },
        {
            "delay_s": 0.5,
            "agent": "system",
            "message": "🗃️ Board Memory: 3 similar past case(s) retrieved (top similarity: 93%)",
            "log_line": "[system] (34%) Board Memory: 3 similar cases retrieved",
        },
        {
            "delay_s": 0.5,
            "agent": "vlm_pathologist",
            "message": "👁️ Qwen2.5-VL-7B: requesting visual second opinion on 4 patches...",
            "log_line": "[vlm_pathologist] (40%) Qwen2.5-VL-7B: requesting visual second opinion...",
        },
        {
            "delay_s": 5.0,
            "agent": "vlm_pathologist",
            "message": (
                "👁️ Qwen2.5-VL: 'lung adenocarcinoma' — Irregular glandular structures "
                "with nuclear atypia, increased N/C ratio, stromal desmoplasia..."
            ),
            "log_line": "[vlm_pathologist] (42%) Qwen2.5-VL: 'lung adenocarcinoma' confirmed",
        },
        {
            "delay_s": 0.8,
            "agent": "vlm_pathologist",
            "message": "👁️ Malignancy indicators: nuclear atypia, irregular gland borders, high N/C ratio, stromal desmoplasia",
            "log_line": "[vlm_pathologist] (43%) Malignancy indicators: nuclear atypia, gland irregularity, high N/C ratio",
        },
        {
            "delay_s": 0.7,
            "agent": "system",
            "message": "🔗 VLM reconciliation: agreement=88/100 · consensus_tissue='lung_adenocarcinoma'",
            "log_line": "[system] (45%) VLM reconciliation: agreement=88/100",
        },
        {
            "delay_s": 0.5,
            "agent": "researcher",
            "message": "📚 Querying Qdrant in-process corpus — retrieving top-8 chunks",
            "log_line": "[researcher] (35%) Querying Qdrant — top-8 chunks retrieved",
        },
        {
            "delay_s": 2.5,
            "agent": "researcher",
            "message": "📚 Evidence loaded: NCCN NSCLC 2024 + 7 TCGA studies — synthesising via Llama 3.3 70B",
            "log_line": "[researcher] (52%) Evidence: NCCN NSCLC 2024 + 7 TCGA studies loaded",
        },
        {
            "delay_s": 1.0,
            "agent": "researcher",
            "message": "📚 Synthesised 4 treatment options (evidence quality: high)",
            "log_line": "[researcher] (56%) Synthesised 4 treatment options",
        },
        {
            "delay_s": 0.5,
            "agent": "tnm_specialist",
            "message": "🏷️ Llama-3.1-8B LoRA: running TNM staging specialist...",
            "log_line": "[tnm_specialist] (57%) Llama-3.1-8B LoRA: running TNM staging...",
        },
        {
            "delay_s": 1.5,
            "agent": "tnm_specialist",
            "message": "🏷️ TNM result: T2bN2M0 · AJCC Stage IIIA — T:T2b  N:N2  M:M0 (confidence: 0.87)",
            "log_line": "[tnm_specialist] (59%) T2bN2M0 → Stage IIIA (AJCC 8th Ed.)",
        },
        {
            "delay_s": 0.5,
            "agent": "biomarker_specialist",
            "message": "🧬 Biomarker specialist: EGFR/ALK/ROS1/PD-L1/KRAS/BRAF/MET panel required (confidence: 0.91)",
            "log_line": "[biomarker_specialist] (60%) EGFR/ALK/ROS1/PD-L1/KRAS/BRAF/MET panel required",
        },
        {
            "delay_s": 0.8,
            "agent": "differential",
            "message": "📋 Primary: Lung Adenocarcinoma (89%) | DDx: Mucinous adenocarcinoma (7%), Large cell carcinoma (4%)",
            "log_line": "[differential] (62%) Lung Adenocarcinoma 89% | Mucinous adeno 7% | LCC 4%",
        },
        {
            "delay_s": 0.7,
            "agent": "treatment_specialist",
            "message": "💊 NCCN Category 1: treatment deferred pending EGFR molecular confirmation",
            "log_line": "[treatment_specialist] (64%) NCCN Cat 1: treatment pending EGFR confirmation",
        },
        {
            "delay_s": 0.5,
            "agent": "oncologist",
            "message": "👨‍⚕️ Llama 3.3 70B: synthesising initial management plan...",
            "log_line": "[oncologist] (65%) Llama 3.3 70B: synthesising management plan...",
        },
        {
            "delay_s": 4.0,
            "agent": "oncologist",
            "message": "👨‍⚕️ Initial plan complete — Lung Adenocarcinoma (confidence: 87%)",
            "log_line": "[oncologist] (72%) Initial plan complete — Lung Adenocarcinoma (87%)",
        },
        {
            "delay_s": 0.5,
            "agent": "system",
            "message": "🗣️ Agent Debate: initiating multi-round deliberation...",
            "log_line": "[system] (74%) Agent Debate: initiating multi-round deliberation",
        },
        {
            "delay_s": 0.5,
            "agent": "researcher",
            "message": "🗣️ Round 1: reviewing draft plan against NCCN guidelines...",
            "log_line": "[researcher] (75%) Round 1: reviewing draft plan vs NCCN guidelines",
        },
        {
            "delay_s": 2.0,
            "agent": "researcher",
            "message": "⚠️ CHALLENGE: EGFR status unknown — NCCN Category 1 TKI requires molecular confirmation before initiation",
            "log_line": "[researcher] (77%) CHALLENGE: EGFR unknown — NCCN Cat 1 pending molecular confirmation",
        },
        {
            "delay_s": 0.5,
            "agent": "oncologist",
            "message": "✏️ Round 1: revising management plan based on challenge...",
            "log_line": "[oncologist] (82%) Round 1: revising plan based on researcher challenge",
        },
        {
            "delay_s": 2.0,
            "agent": "oncologist",
            "message": "✏️ Revision accepted: molecular panel added to immediate actions — consensus improving",
            "log_line": "[oncologist] (84%) Revision: molecular panel added — consensus improving",
        },
        {
            "delay_s": 1.0,
            "agent": "system",
            "message": "✅ Consensus score: 87/100 — consensus reached, debate complete",
            "log_line": "[system] (89%) Consensus 87/100 — debate complete",
        },
        {
            "delay_s": 0.5,
            "agent": "patient_summary",
            "message": "📄 Patient summary ready (plain English, 8th-grade reading level)",
            "log_line": "[patient_summary] (89%) Patient summary ready (8th-grade English)",
        },
        {
            "delay_s": 0.5,
            "agent": "trial_matcher",
            "message": "🔍 2 potentially eligible clinical trial(s) found (NCT05261399, NCT04667234)",
            "log_line": "[trial_matcher] (91%) 2 eligible trials found",
        },
        {
            "delay_s": 0.5,
            "agent": "system",
            "message": "📈 Digital Twin: 12-month PFS prediction 78% ± 6% (TCGA LUAD kinetics)",
            "log_line": "[system] (92%) Digital Twin: 12-month PFS 78% ± 6%",
        },
        {
            "delay_s": 0.5,
            "agent": "counterfactual",
            "message": "🔄 Counterfactual (EGFR-negative): first-line → Carboplatin + Pemetrexed + Pembrolizumab",
            "log_line": "[counterfactual] (95%) Counterfactual EGFR-: chemo-immuno combination",
        },
        {
            "delay_s": 0.5,
            "agent": "system",
            "message": "✅ Analysis complete — Lung Adenocarcinoma | 1 debate round completed | ~35s",
            "log_line": "[system] (100%) Analysis complete — 1 debate round | ~35s",
        },
    ]


def _mock_report(case_name: str) -> dict:
    """Return a rich mock management plan for demo/offline mode."""
    tissue_label = case_name.replace("_", " ").title()
    return {
        "case_id": f"demo_{case_name}",
        "status": "done",
        "total_time_s": 34.2,
        "pathology_report": {
            "tissue_type": case_name,
            "confidence": 0.942,
            "uncertainty_interval": "94.2% ± 3.1%",
            "high_uncertainty": False,
            "flags": ["glandular_pattern", "nuclear_atypia", "lepidic_growth"],
            "n_patches": 12,
            "biomarkers": {
                "EGFR": {"score": 0.72, "level": "high"},
                "ALK":  {"score": 0.18, "level": "low"},
                "PD_L1":{"score": 0.55, "level": "moderate"},
            },
        },
        "management_plan": {
            "diagnosis": {
                "primary": tissue_label,
                "tnm_stage": "T2bN2M0 — Stage IIIA (AJCC 8th Ed.)",
                "confidence": 0.942,
            },
            "immediate_actions": [
                "Order comprehensive molecular panel: EGFR/ALK/ROS1/KRAS/PD-L1/BRAF/MET/RET",
                "Staging CT chest/abdomen/pelvis with contrast",
                "Brain MRI with gadolinium",
                "Multidisciplinary tumour board referral for surgical candidacy",
            ],
            "treatment_plan": {
                "first_line": (
                    "PENDING MOLECULAR RESULTS — do not initiate targeted therapy before "
                    "biomarker results. If EGFR+: Osimertinib 80 mg OD (NCCN Category 1). "
                    "If PD-L1 ≥50% & driver-negative: Pembrolizumab 200 mg Q3W (NCCN Category 1)."
                ),
                "rationale": (
                    "NCCN 2024 mandates comprehensive molecular testing before first-line systemic "
                    "therapy in adenocarcinoma. TKI in EGFR-mutant disease yields superior PFS "
                    "vs chemotherapy (FLAURA2 2023)."
                ),
                "alternatives": [
                    "Carboplatin + Pemetrexed + Pembrolizumab (driver-negative, PD-L1<50%)",
                    "Clinical trial enrollment (NCCN preferred if available)",
                    "Definitive chemoRT → Durvalumab consolidation (PACIFIC, if unresectable)",
                ],
            },
            "debate_rounds_completed": 1,
            "consensus_score": 87,
            "board_consensus": (
                "High consensus (87/100) after 1 debate round. Primary revision: "
                "treatment deferred pending mandatory molecular testing — "
                "consensus improved from 71 → 87 after amendment."
            ),
            "citations": [
                "Wu YL et al. FLAURA2. NEJM 2023;389:645–657",
                "Reck M et al. KEYNOTE-024. NEJM 2018;379:2040–2051",
                "Gandhi L et al. KEYNOTE-189. NEJM 2018;378:2078–2092",
                "Antonia SJ et al. PACIFIC. NEJM 2018;379:2342–2350",
                "NCCN NSCLC Guidelines v4.2024",
            ],
            "pfs_12mo": 0.78,
            "disclaimer": "AI research tool. NOT for clinical use. Requires oncologist review.",
        },
        "debate_transcript": [
            {
                "round": 1,
                "speaker": "researcher",
                "message": (
                    "⚠️ CHALLENGE: Treatment plan proposes TKI initiation without confirmed EGFR status. "
                    "NCCN Category 1 evidence for osimertinib is conditional on molecular confirmation."
                ),
            },
            {
                "round": 1,
                "speaker": "oncologist",
                "message": (
                    "Revision accepted. Treatment initiation explicitly deferred pending molecular panel. "
                    "Mandatory testing added to immediate actions with 7-day reconvene."
                ),
                "consensus_score": 87,
            },
        ],
    }


# ── Core functions ─────────────────────────────────────────────────────────────

def run_demo_case(case_name: str):
    """
    Run a pre-baked demo case through the AOB pipeline.

    This is a generator so Gradio streams intermediate status updates to the UI.
    Yields: (status_text, log_text, report_json) tuples progressively.
    In mock mode: simulates ~33 s pipeline with time.sleep() between stages.
    In live mode: polls /status every POLL_INTERVAL seconds until done.
    """
    if not API_URL:
        # ── Mock mode: animate stages with sleep ──────────────────────────────
        log_lines: list[str] = [f"[mock] Running '{case_name}' in demo mode (AOB_API_URL not set)"]
        yield "⏳ Starting mock pipeline...", "\n".join(log_lines), None

        for stage in _mock_stages():
            time.sleep(stage["delay_s"])
            log_lines.append(stage["log_line"])
            yield stage["message"], "\n".join(log_lines), None

        report = _mock_report(case_name)
        consensus = report.get("management_plan", {}).get("consensus_score", 87)
        yield (
            f"✅ Complete (mock) — 1 debate round | consensus: {consensus}/100 | ~34s",
            "\n".join(log_lines),
            report,
        )
        return

    # ── Live mode: submit + poll ───────────────────────────────────────────────
    resp = _api_post(f"/demo/run/{case_name}", {})
    if not resp:
        yield "❌ Failed to submit demo case", "", None
        return

    job_id = resp.get("job_id", "")
    if not job_id:
        yield "❌ No job_id in response", "", None
        return

    t0 = time.time()
    log_lines = [f"Job {job_id} submitted..."]
    yield f"⏳ Job {job_id} submitted...", log_lines[0], None

    while time.time() - t0 < MAX_WAIT_S:
        time.sleep(POLL_INTERVAL)
        status_data = _api_get(f"/status/{job_id}")
        if status_data:
            steps = status_data.get("steps", [])
            new_steps = steps[len(log_lines) - 1:]
            for step in new_steps:
                agent = step.get("agent", "system")
                msg   = step.get("message", "")
                prog  = step.get("progress", 0)
                log_lines.append(f"[{agent}] ({prog}%) {msg}")

            job_status = status_data.get("status", "")
            last_msg = log_lines[-1] if log_lines else "Running..."
            yield f"⏳ {last_msg}", "\n".join(log_lines), None

            if job_status == "done":
                break
            if job_status == "failed":
                err = status_data.get("error", "Unknown error")
                yield f"❌ Pipeline failed: {err}", "\n".join(log_lines), None
                return

    report_data = _api_get(f"/report/{job_id}")
    if not report_data:
        yield "❌ Could not fetch report", "\n".join(log_lines), None
        return

    mp = report_data.get("management_plan", {})
    status_text = (
        f"✅ Complete — {mp.get('debate_rounds_completed', 0)} debate round(s) | "
        f"consensus: {mp.get('consensus_score', '?')}/100 | "
        f"Time: {report_data.get('total_time_s', '?')}s"
    )
    yield status_text, "\n".join(log_lines), report_data


def run_custom_case(
    images: list,
    patient_age: int,
    sex: str,
    clinical_notes: str,
    egfr_status: str,
    alk_status: str,
):
    """
    Run the pipeline on uploaded image patches.

    Generator: yields (status_text, log_text, report_json) progressively.
    In mock mode: simulates the full pipeline with time.sleep() between stages.
    """
    if not images:
        yield "❌ Please upload at least one image patch", "", None
        return

    if not API_URL:
        # ── Mock mode for custom upload ───────────────────────────────────────
        n = len(images)
        log_lines: list[str] = [
            f"[mock] {n} patch(es) received — running demo mode (AOB_API_URL not set)",
            f"[mock] Patient: {patient_age}y {sex} | EGFR: {egfr_status} | ALK: {alk_status}",
        ]
        yield "⏳ Starting mock pipeline for uploaded patches...", "\n".join(log_lines), None

        for stage in _mock_stages():
            time.sleep(stage["delay_s"])
            log_lines.append(stage["log_line"])
            yield stage["message"], "\n".join(log_lines), None

        report = _mock_report("custom_upload")
        # Override biomarker status in the report if user provided values
        try:
            bm = report["pathology_report"]["biomarkers"]
            if egfr_status != "unknown":
                bm["EGFR"]["level"] = egfr_status
            if alk_status != "unknown":
                bm["ALK"]["level"] = alk_status
        except (KeyError, TypeError):
            pass
        consensus = report.get("management_plan", {}).get("consensus_score", 87)
        yield (
            f"✅ Complete (mock) — 1 debate round | consensus: {consensus}/100 | ~34s",
            "\n".join(log_lines),
            report,
        )
        return

    # ── Live mode ─────────────────────────────────────────────────────────────
    patches_b64: list[str] = []
    for img in images:
        try:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            patches_b64.append(base64.b64encode(buf.getvalue()).decode())
        except Exception as e:
            yield f"❌ Failed to encode image: {e}", "", None
            return

    payload = {
        "patches_b64": patches_b64,
        "metadata": {
            "patient_age": patient_age,
            "sex": sex,
            "clinical_notes": clinical_notes,
            "biomarker_status": {"EGFR": egfr_status, "ALK": alk_status},
        },
    }

    resp = _api_post("/analyze", payload)
    if not resp:
        yield "❌ Failed to submit case to API", "", None
        return

    job_id = resp.get("job_id", "")
    t0 = time.time()
    log_lines = [f"Job {job_id} submitted ({len(patches_b64)} patches)..."]
    yield f"⏳ Job {job_id} submitted...", log_lines[0], None

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
            last_msg = log_lines[-1] if log_lines else "Running..."
            yield f"⏳ {last_msg}", "\n".join(log_lines), None

            if job_status == "done":
                break
            if job_status == "failed":
                err = status_data.get("error", "Unknown error")
                yield f"❌ Pipeline failed: {err}", "\n".join(log_lines), None
                return

    report_data = _api_get(f"/report/{job_id}")
    if not report_data:
        yield "❌ Could not fetch report", "\n".join(log_lines), None
        return

    mp = report_data.get("management_plan", {})
    status_text = (
        f"✅ Complete — {mp.get('debate_rounds_completed', 0)} debate round(s) | "
        f"consensus: {mp.get('consensus_score', '?')}/100 | "
        f"Time: {report_data.get('total_time_s', '?')}s"
    )
    yield status_text, "\n".join(log_lines), report_data


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
                        egfr_input  = gr.Radio(["unknown", "positive", "negative"], value="unknown", label="EGFR Status")
                        alk_input   = gr.Radio(["unknown", "positive", "negative"], value="unknown", label="ALK Status")
                        run_btn     = gr.Button("▶️ Analyse Case", variant="primary")

                    with gr.Column():
                        upload_status = gr.Textbox(label="Status", interactive=False)
                        upload_log    = gr.Textbox(label="Agent Pipeline Log", lines=12, interactive=False)

                upload_report = gr.JSON(label="Management Plan (JSON)")

                run_btn.click(
                    fn=run_custom_case,
                    inputs=[img_upload, age_input, sex_input, notes_input, egfr_input, alk_input],
                    outputs=[upload_status, upload_log, upload_report],
                )

            # ── Tab 3: About ────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## How It Works

                The Autonomous Oncology Board (AOB) simulates a multi-disciplinary tumour board meeting:

                1. **Agent 1 — Pathologist** (Prov-GigaPath ViT-Giant)
                   - Analyses histopathology image patches
                   - Returns tissue type, confidence, and biomarker-related signals
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
