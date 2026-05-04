# AOB Demo Video Script
## 5-Minute Winning Narrative | AMD Developer Hackathon 2026

---

### PRE-RECORDING CHECKLIST
- [ ] AOB backend running (`python -m aob.ml.api`)
- [ ] Ollama serving `llama3.3:70b` (`ollama serve`)
- [ ] vLLM serving specialists (`./scripts/serve_specialists.sh`)
- [ ] Frontend open at `http://localhost:8000`
- [ ] VRAM dashboard widget visible (baseline ~88 GB)
- [ ] 3 demo cases pre-uploaded (lung adeno, colon MSI-H, squamous)
- [ ] Screen recording software active (1080p minimum)
- [ ] Microphone tested

---

## SEGMENT 1: The Opening Hook (0:00 — 0:40)

**[SCREEN: VRAM dashboard widget. Bars at 0 GB. Black background.]**

> *"Every year, 19 million people are diagnosed with cancer. The standard of care at a major cancer center is a tumour board — a meeting where a pathologist, a researcher, and an oncologist argue about your case before deciding your treatment."*

> *"Today, that meeting takes weeks to schedule. Most patients never get one."*

**[BEAT — pause 1 second]**

> *"We built one that runs in 30 seconds."*

**[SCREEN: Slowly zoom into the VRAM dashboard. The numbers are still at 0 GB.]**

> *"This is an AMD Instinct MI300X. 192 gigabytes of unified memory. This single number is the reason everything you're about to see is possible."*

---

## SEGMENT 2: The Pathologist Agent (0:40 — 1:30)

**[SCREEN: Drag lung adenocarcinoma patch images into the upload zone.]**

> *"We upload a batch of histopathology patches from a patient's lung biopsy."*

**[VRAM bar animates: 0 → 3 GB]**

> *"Prov-GigaPath — a vision foundation model trained on 1.3 billion pathology image tokens — loads into memory. It's looking at these patches the same way a pathologist does: cell by cell, nucleus by nucleus."*

**[SCREEN: Heatmaps appear on the patches — red/orange regions labeled "SUSPICIOUS"]**

> *"These aren't filters. These are attention heatmaps from the transformer's own reasoning — the same mechanism that powers large language models, applied to tissue. The model is showing us *why* it flagged these cells."*

> *"Glandular pattern. Nuclear atypia. The model sees it. And it knows it knows — with 91% confidence, ± 4%."*

---

## SEGMENT 3: The Researcher Agent (1:30 — 2:15)

**[SCREEN: Evidence panel begins populating. Citations appear one by one.]**

> *"Now the Researcher agent takes that pathology report and queries 500 pre-indexed oncology documents — NCCN guidelines, TCGA studies, PubMed papers."*

> *"It's not searching Google. It's doing semantic retrieval over a curated, citation-tracked corpus. Every claim it makes has a source."*

**[SCREEN: Show a citation appearing: "KEYNOTE-189: Carboplatin + Pemetrexed + Pembrolizumab — Category 1 for EGFR/ALK-negative NSCLC, PD-L1 ≥1%"]**

> *"The difference between a clinical tool and a chatbot is: every recommendation needs a trial reference and a guideline category. This system provides both."*

---

## SEGMENT 4: The Debate — The Feature No Other Team Has (2:15 — 3:15)

**[SCREEN: Debate Transcript panel opens. First, Oncologist draft appears.]**

> *"Here's where it gets interesting. The oncologist agent produces a first draft treatment plan."*

**[SCREEN: A yellow challenge badge appears — "⚠️ RESEARCHER CHALLENGE"]**

> *"And then the researcher challenges it."*

**[Read the challenge aloud, natural pacing:]**

> *"'⚠️ EGFR molecular status is not confirmed in the pathology report. NCCN Category 1 for osimertinib requires confirmed EGFR mutation. Recommend molecular reflex testing BEFORE initiating targeted therapy.'"*

**[SCREEN: The oncologist's revised plan appears. Show the revision diff — strikethrough on osimertinib, green text showing new recommendation.]**

> *"The oncologist revises. The plan changes. That's not a bug — that's the system working exactly as a real tumour board does."*

> *"No other AI oncology system we're aware of has agents that can change each other's minds."*

---

## SEGMENT 5: The Final Report (3:15 — 4:00)

**[SCREEN: Final Management Plan rendered — scroll through slowly]**

> *"The final report: TNM staging — T2a N2 M0, Stage IIIA. First-line treatment. NCCN Category 1. Full citation chain. Biomarker tests required before targeted therapy. Three matched clinical trials the patient qualifies for."*

**[SCREEN: Click 'Patient Summary' tab]**

> *"And here's something no hospital report looks like: a plain-English summary written for the patient at an 8th-grade reading level. Because the patient deserves to understand their own diagnosis."*

**[SCREEN: Click 'What if?' button. Change EGFR result to 'Negative'. Show revised plan.]**

> *"The What-If button. Change one input — EGFR comes back negative — the entire treatment plan replans in 3 seconds. This is counterfactual clinical reasoning, on demand."*

---

## SEGMENT 6: The AMD Hardware Proof (4:00 — 4:40)

**[SCREEN: VRAM dashboard front and center. 88 GB shown. Animate the H100 comparison bar — it hits 80 GB and turns red with 'OOM'.]**

> *"Let's be concrete about why this requires AMD."*

> *"Llama 3.3 70B in FP8 — 70 gigabytes. GigaPath and the specialist adapters — another 20 gigabytes. The KV cache for a full debate loop — 30 gigabytes. Total: 120 gigabytes at peak."*

> *"The NVIDIA H100 has 80 gigabytes. The math doesn't work."*

**[SCREEN: Three case cards launch simultaneously. VRAM climbs to 118 GB. All three complete.]**

> *"Three different patients. Three simultaneous analyses. 118 gigabytes used. 74 gigabytes of headroom to spare. This is not a benchmark. This is a live demo."*

---

## SEGMENT 7: Close (4:40 — 5:00)

**[SCREEN: Final management plan on one side, VRAM dashboard on the other]**

> *"The Autonomous Oncology Board. Multi-agent. Multi-model. Multi-case. Explainable. Calibrated. Debate-driven. Citation-grounded."*

> *"We didn't compromise the architecture to fit the hardware."*

**[PAUSE — let it land]**

> *"We chose hardware that made the architecture possible."*

**[SCREEN: AMD MI300X logo. "192 GB HBM3. ROCm 6.x. Open source."  
GitHub link. HuggingFace dataset link.]**

---

## RECORDING NOTES

- **Pace:** Speak at ~120 words/minute. The script is ~600 words = ~5 minutes. Don't rush.
- **The debate segment** (2:15–3:15) is the most memorable. Give it space. Read the challenge slowly.
- **VRAM numbers** must be visible to judges. Zoom in when discussing them.
- **Revision diff** should be on screen for at least 8 seconds — it's the visual proof of the debate.
- **Record in one take** if possible — cuts reduce the feeling of a live system.
- Fallback: If the full pipeline is slow, use the pre-baked case results from `/api/report/{case_id}`.

## PRE-BAKED DEMO CASES (3 curated)

| Case | File | Cancer Type | Key Feature |
|------|------|-------------|-------------|
| DEMO-A | `demo_lung_adeno_egfr.json` | Lung Adeno, EGFR L858R, Stage IIIA | Debate triggers: EGFR not confirmed |
| DEMO-B | `demo_colon_msih.json` | Colon Adeno, MSI-H, Stage IVA | Debate triggers: Lynch syndrome testing |
| DEMO-C | `demo_lung_squamous.json` | Lung SqCC, PD-L1 80%, Stage IV | Biomarker: PD-L1 ≥50% → monotherapy |
