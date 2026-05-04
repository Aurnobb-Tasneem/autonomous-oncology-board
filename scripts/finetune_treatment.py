"""
scripts/finetune_treatment.py
==============================
LoRA fine-tuning of Llama-3.1-8B-Instruct for oncology treatment plan
generation on AMD MI300X.

Track 2 (Fine-Tuning on AMD GPUs) deliverable — Treatment Specialist.

What this does
--------------
1. Creates 50 examples mapping (pathology + stage + biomarkers) → structured
   treatment plan JSON.
2. Trains LoRA on meta-llama/Llama-3.1-8B-Instruct via the generic lora_trainer.
3. Saves the adapter to <output_dir>/ with training_report.json.
4. Served alongside tnm_specialist + biomarker_specialist via vLLM.

Output JSON schema:
    {
      "first_line": "...",
      "second_line": "...",
      "nccn_category": "1" | "2A" | "2B",
      "contraindications": [...],
      "monitoring": [...]
    }

Usage
-----
    export HF_TOKEN=hf_...
    python scripts/finetune_treatment.py

    # Quick smoke (50 steps):
    python scripts/finetune_treatment.py --max_steps 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("finetune_treatment")

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified oncologist and NCCN guideline expert.
Given a clinical case description (tissue type, TNM stage, and biomarker status),
output ONLY a JSON object with keys:
  first_line (recommended first-line treatment regimen as a string),
  second_line (recommended second-line regimen as a string),
  nccn_category (NCCN evidence category: "1", "2A", or "2B"),
  contraindications (list of contraindications to first-line therapy),
  monitoring (list of monitoring parameters during treatment).
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

OUTPUT_SCHEMA_KEYS = {"first_line", "second_line", "nccn_category", "contraindications", "monitoring"}

# ── 50-example training dataset ───────────────────────────────────────────────
_TREATMENT_EXAMPLES: list[tuple[str, dict]] = [
    # ── Lung Adenocarcinoma — driver-positive (12 examples) ──────────────────
    (
        "Lung adenocarcinoma. Stage IV (T2a N2 M1b, liver). EGFR exon 19 deletion. ALK/ROS1 negative. PD-L1 TPS 30%.",
        {
            "first_line":       "Osimertinib 80 mg PO daily",
            "second_line":      "Platinum doublet (carboplatin + pemetrexed) + bevacizumab on osimertinib progression",
            "nccn_category":    "1",
            "contraindications": ["QTc prolongation >470 ms (ECG baseline required)", "Severe hepatic impairment (Child-Pugh C)", "Interstitial lung disease"],
            "monitoring":       ["ECG at baseline and monthly (QTc)", "LFTs q3 months", "HRCT chest if new respiratory symptoms", "ctDNA T790M/C797S at progression"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IIIB (T4 N2 M0). EGFR L858R. Unresectable. No prior systemic therapy.",
        {
            "first_line":       "Osimertinib 80 mg PO daily (TKI preferred over concurrent CRT + consolidation in EGFR-mutant)",
            "second_line":      "Carboplatin + pemetrexed + bevacizumab (post-osimertinib progression)",
            "nccn_category":    "1",
            "contraindications": ["Active haemoptysis (relative, assess per clinical context)", "Severe renal impairment (pemetrexed dose adjustment required)"],
            "monitoring":       ["CT chest q6-8 weeks initially", "EGFR resistance panel (liquid biopsy) at progression", "LFTs q3 months"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. ALK rearrangement confirmed. No prior targeted therapy.",
        {
            "first_line":       "Alectinib 600 mg PO BID with food",
            "second_line":      "Brigatinib 90 mg PO daily × 7 days then 180 mg PO daily (ALK progression); lorlatinib if G1202R mutation",
            "nccn_category":    "1",
            "contraindications": ["Bradycardia <40 bpm (discontinue until recovery)", "Severe hepatic impairment"],
            "monitoring":       ["LFTs q2 weeks for first 3 months, then q3 months", "ECG at baseline (bradycardia risk)", "CPK at baseline (myositis rare but reported)", "Ophthalmology consult if visual changes"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. ROS1 fusion (ROS1-CD74). Brain metastases × 2.",
        {
            "first_line":       "Entrectinib 600 mg PO daily (CNS-penetrant, NCCN preferred for brain mets)",
            "second_line":      "Lorlatinib 100 mg PO daily (after entrectinib/crizotinib failure)",
            "nccn_category":    "1",
            "contraindications": ["Moderate-severe hepatic impairment (dose reduction required)", "Concomitant strong CYP3A4 inhibitors (increase entrectinib exposure)"],
            "monitoring":       ["MRI brain q8-12 weeks (CNS disease)", "LFTs monthly for first 6 months", "ECG baseline (QTc)", "Weight monitoring (entrectinib CNS toxicity)"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. KRAS G12C mutation. EGFR/ALK/ROS1 negative. PD-L1 TPS 10%. Post-platinum+pembrolizumab.",
        {
            "first_line":       "Sotorasib 960 mg PO daily (KRAS G12C, second-line post-platinum/IO)",
            "second_line":      "Adagrasib 400 mg PO BID (KRAS G12C, if sotorasib intolerance or resistance)",
            "nccn_category":    "1",
            "contraindications": ["Severe hepatic impairment (ALT/AST >3× ULN at baseline)", "Active ILD/pneumonitis"],
            "monitoring":       ["LFTs q2 weeks for 3 months, then q3 months (hepatotoxicity)", "CT chest at 6-8 weeks", "Renal function (adagrasib: QTc prolongation risk)"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. MET exon 14 skipping mutation confirmed.",
        {
            "first_line":       "Tepotinib 500 mg PO daily with food",
            "second_line":      "Capmatinib 400 mg PO BID (tepotinib intolerance or progression)",
            "nccn_category":    "1",
            "contraindications": ["Severe peripheral oedema (MET inhibitors can worsen fluid retention)", "Strong CYP3A inducers (reduce tepotinib levels)"],
            "monitoring":       ["Peripheral oedema assessment weekly for 3 months", "LFTs monthly", "CT chest q8 weeks response assessment"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. BRAF V600E mutation. PD-L1 TPS 5%. No prior targeted therapy.",
        {
            "first_line":       "Dabrafenib 150 mg PO BID + trametinib 2 mg PO daily",
            "second_line":      "Platinum + pemetrexed + pembrolizumab (BRAF-targeted progression)",
            "nccn_category":    "2A",
            "contraindications": ["Severe left ventricular dysfunction (LVEF <50%)", "Active RAS-mutated malignancy (BRAF inhibitor paradoxical activation)", "Vemurafenib + cobimetinib not preferred (inferior CNS penetrance)"],
            "monitoring":       ["Echo at baseline, 1 month, then q3 months (LVEF)", "Dermatology review q3 months (cuSCC risk)", "Blood glucose (trametinib hyperglycaemia)", "CT q8 weeks"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. NTRK3 fusion. Treatment-naive.",
        {
            "first_line":       "Larotrectinib 100 mg PO BID",
            "second_line":      "Entrectinib 600 mg PO daily (NTRK, if larotrectinib intolerance)",
            "nccn_category":    "1",
            "contraindications": ["Strong CYP3A4 inhibitors (increase larotrectinib AUC significantly)", "Moderate-severe hepatic impairment (dose reduction required)"],
            "monitoring":       ["LFTs q2 weeks for 6 months, then monthly", "Neurotoxicity assessment (dizziness, gait disturbance)", "CT q8-12 weeks response assessment"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. HER2 exon 20 insertion. Post-platinum failure.",
        {
            "first_line":       "Trastuzumab deruxtecan (T-DXd) 5.4 mg/kg IV q3w",
            "second_line":      "Mobocertinib 160 mg PO daily (HER2 exon 20 ins, alternative post-T-DXd)",
            "nccn_category":    "2A",
            "contraindications": ["Interstitial lung disease / pneumonitis history (ILD risk with T-DXd)", "LVEF <50% (trastuzumab cardiac toxicity)", "Pregnancy"],
            "monitoring":       ["LVEF at baseline and q3 months", "Chest CT for early ILD detection (baseline and q6 weeks)", "CBC weekly for first cycle (cytopenias)"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. PD-L1 TPS 80%. No actionable driver mutations. Treatment-naive.",
        {
            "first_line":       "Pembrolizumab 200 mg IV q3w (PD-L1 ≥50%, monotherapy)",
            "second_line":      "Carboplatin + pemetrexed + bevacizumab (post-pembrolizumab progression)",
            "nccn_category":    "1",
            "contraindications": ["Active autoimmune disease requiring systemic steroids", "Prior organ transplant", "Uncontrolled thyroid disorder"],
            "monitoring":       ["TFTs q6 weeks (immune-related thyroiditis)", "LFTs q3 weeks (immune hepatitis)", "Glucose q3 weeks (immune endocrinopathy)", "Prompt CT if new respiratory symptoms (pneumonitis)"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IB post-resection (R0). EGFR exon 19 deletion. pT2b N0 M0.",
        {
            "first_line":       "Osimertinib 80 mg PO daily × 3 years adjuvant (ADAURA trial, NCCN Category 1 Stage IB-IIIA)",
            "second_line":      "Observe after adjuvant completion; rebiopsy at recurrence for resistance profiling",
            "nccn_category":    "1",
            "contraindications": ["Active ILD", "QTc prolongation", "Deferred if PS ≥3 (adjuvant intent requires PS 0-2)"],
            "monitoring":       ["CT chest q6 months for 2 years, then annual", "ECG at 3 and 6 months adjuvant", "LFTs q3 months"],
        }
    ),
    (
        "Lung adenocarcinoma. Stage IV. EGFR exon 20 insertion (S768_D770dup). No prior targeted therapy.",
        {
            "first_line":       "Amivantamab-vmjw 1050 mg IV qw × 4 doses, then q2w (EGFR exon 20 ins, post-platinum — PAPILLON data Category 2A first-line)",
            "second_line":      "Platinum + pemetrexed (if amivantamab not accessible first-line)",
            "nccn_category":    "2A",
            "contraindications": ["Severe infusion reactions (premedicate per protocol)", "EGFR exon 19/21 (not exon 20 ins — different agent)"],
            "monitoring":       ["Infusion reaction monitoring (first 3 infusions)", "Skin toxicity assessment (EGFR-class rash)", "LFTs q3 weeks", "CT q8 weeks"],
        }
    ),

    # ── Lung Squamous Cell Carcinoma (10 examples) ───────────────────────────
    (
        "Lung squamous cell carcinoma. Stage IV (T3 N3 M1b). PD-L1 TPS 60%. No driver mutations.",
        {
            "first_line":       "Pembrolizumab 200 mg IV q3w (PD-L1 ≥50%, monotherapy, KEYNOTE-024)",
            "second_line":      "Carboplatin + paclitaxel + pembrolizumab (post-IO progression)",
            "nccn_category":    "1",
            "contraindications": ["Active autoimmune disease", "Active ILD", "Uncontrolled diabetes (immune endocrinopathy risk)"],
            "monitoring":       ["TFTs q6 weeks", "LFTs q3 weeks", "HbA1c every 3 cycles", "CT q8 weeks"],
        }
    ),
    (
        "Lung squamous cell carcinoma. Stage IV. PD-L1 TPS 30%. No driver mutations. Treatment-naive.",
        {
            "first_line":       "Carboplatin AUC5 + paclitaxel 200 mg/m² + pembrolizumab 200 mg IV q3w × 4 cycles, then pembrolizumab maintenance",
            "second_line":      "Docetaxel 75 mg/m² + ramucirumab 10 mg/kg q3w (post-platinum/IO)",
            "nccn_category":    "1",
            "contraindications": ["Peripheral neuropathy ≥Grade 2 (paclitaxel)", "Active bleeding (ramucirumab contraindicated with haemoptysis)", "ECOG PS ≥3"],
            "monitoring":       ["CBC q3 weeks (neutropenia)", "Peripheral neuropathy assessment", "LFTs q3 weeks", "ECG (pembrolizumab myocarditis)"],
        }
    ),
    (
        "Lung squamous carcinoma. Stage IIIA (T2b N2 M0). Unresectable. No prior treatment.",
        {
            "first_line":       "Concurrent platinum-based CRT (cisplatin 50 mg/m² days 1,8,29,36 + etoposide, concurrent with 60 Gy RT) then durvalumab 10 mg/kg q2w × 12 months",
            "second_line":      "Docetaxel (post-CRT/durvalumab progression if resection not feasible)",
            "nccn_category":    "1",
            "contraindications": ["Severe COPD (FEV1 <40% predicted — RT field planning required)", "Active autoimmune disease (durvalumab)", "Severe cardiac disease (cisplatin cumulative toxicity)"],
            "monitoring":       ["PFTs before/after CRT", "TFTs q6 weeks (durvalumab thyroiditis)", "HRCT chest for radiation pneumonitis (baseline, q3 months post-CRT)"],
        }
    ),
    (
        "Squamous cell lung carcinoma. Stage IIA post-resection. PD-L1 0%. Negative nodes. No adjuvant chemo given.",
        {
            "first_line":       "Observation with CT surveillance (Stage IIA R0 squamous, no adjuvant chemo standard if margins clear, high-risk features absent)",
            "second_line":      "Carboplatin + gemcitabine (if recurrence and prior CT-naive)",
            "nccn_category":    "2A",
            "contraindications": ["Adjuvant osimertinib not indicated (squamous, not EGFR-mutant adenocarcinoma)"],
            "monitoring":       ["CT chest q6 months × 2 years, then annually", "Smoking cessation counselling", "PFTs annually"],
        }
    ),
    (
        "Lung squamous carcinoma. Stage IV. Post-platinum failure. PD-L1 TPS 5%. TMB 8 mut/Mb.",
        {
            "first_line":       "Nivolumab 240 mg IV q2w or 480 mg q4w (second-line post-platinum, CheckMate 017)",
            "second_line":      "Docetaxel 75 mg/m² q3w (post-IO progression)",
            "nccn_category":    "1",
            "contraindications": ["Active autoimmune disease", "Systemic corticosteroids >10 mg prednisone equivalent"],
            "monitoring":       ["LFTs q3 weeks", "TFTs monthly", "CBC q3 weeks", "Prompt imaging if new pulmonary symptoms"],
        }
    ),
    (
        "Lung squamous carcinoma. Stage IV. High TMB (22 mut/Mb). PD-L1 TPS 15%.",
        {
            "first_line":       "Nivolumab 360 mg IV q3w + ipilimumab 1 mg/kg q6w (high TMB, CheckMate 227)",
            "second_line":      "Carboplatin + paclitaxel (post-IO progression)",
            "nccn_category":    "2A",
            "contraindications": ["Active autoimmune disease requiring immunosuppression", "Prior Grade ≥3 immune-related adverse event"],
            "monitoring":       ["irAE surveillance: TFTs, LFTs, cortisol, CBC q3 weeks", "Colitis monitoring (ipilimumab)", "CT q8 weeks response"],
        }
    ),
    (
        "Lung squamous carcinoma. Stage IV. Third-line (post-platinum, post-docetaxel, post-nivolumab).",
        {
            "first_line":       "Ramucirumab 10 mg/kg + docetaxel 75 mg/m² q3w (third-line, REVEL trial evidence)",
            "second_line":      "Erlotinib (salvage if EGFR status unknown and no prior TKI — low probability)", 
            "nccn_category":    "2A",
            "contraindications": ["Active haemoptysis ≥Grade 2 (ramucirumab VEGFR2 inhibitor — severe haemoptysis risk in squamous)", "ECOG PS ≥3"],
            "monitoring":       ["CBC weekly × 2 cycles (neutropenia)", "Blood pressure (ramucirumab hypertension)", "Proteinuria q3 weeks (24h urine if 2+ dipstick)"],
        }
    ),
    (
        "Squamous cell lung carcinoma. Stage IIB post-resection. Cisplatin + vinorelbine adjuvant completed × 4 cycles. Now disease-free at 18 months.",
        {
            "first_line":       "CT surveillance (disease-free on adjuvant completion)",
            "second_line":      "Pembrolizumab (recurrence — PD-L1 and MSI-H assessment at recurrence)",
            "nccn_category":    "1",
            "contraindications": ["No active treatment indicated in disease-free surveillance"],
            "monitoring":       ["CT chest/upper abdomen q6 months × 2 years, then annually", "PFTs annually", "PET-CT if CT equivocal finding"],
        }
    ),
    (
        "Squamous cell lung carcinoma. Stage IV. ECOG PS 3. Significant comorbidities.",
        {
            "first_line":       "Pembrolizumab monotherapy 200 mg q3w if PD-L1 ≥50% (PS 3 acceptable for IO monotherapy per NCCN)", 
            "second_line":      "Best supportive care / palliative care if IO not tolerated",
            "nccn_category":    "2A",
            "contraindications": ["Platinum doublet + IO (PS 3 — excess toxicity)", "Aggressive combination regimens"],
            "monitoring":       ["Close toxicity monitoring (weekly clinical review first 2 cycles)", "Palliative care co-management from start", "ECOG PS reassessment q3 weeks"],
        }
    ),
    (
        "Squamous cell lung carcinoma. Superficial, Stage 0 (carcinoma in situ). Bronchoscopic resection planned.",
        {
            "first_line":       "Photodynamic therapy (PDT) or bronchoscopic resection (APC/cryotherapy) — no systemic therapy",
            "second_line":      "Surgical wedge resection if bronchoscopic clearance incomplete",
            "nccn_category":    "1",
            "contraindications": ["Cisplatin/IO — not indicated for Stage 0"],
            "monitoring":       ["Repeat bronchoscopy 6 weeks post-PDT", "CT chest annually", "Surveillance bronchoscopy q6 months × 2 years"],
        }
    ),

    # ── Colon Adenocarcinoma (20 examples) ───────────────────────────────────
    (
        "Colon adenocarcinoma. pT3 N0 M0 (Stage II). MSS. KRAS G12D. High-risk features: T4, perforation — absent. Low risk.",
        {
            "first_line":       "Surveillance (Stage II low-risk, no adjuvant chemo per NCCN in absence of high-risk features)",
            "second_line":      "FOLFOX × 6 months (if high-risk features emerge at MDT review)",
            "nccn_category":    "2A",
            "contraindications": ["Bevacizumab adjuvant — not standard", "FOLFIRI adjuvant — no Stage II evidence"],
            "monitoring":       ["CEA q3-6 months × 3 years", "CT CAP q12 months × 3 years", "Colonoscopy at 1 year"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N2a M0 (Stage IIIB). KRAS wild-type. MSS. Left-sided primary.",
        {
            "first_line":       "FOLFOX (oxaliplatin 85 mg/m² + leucovorin 400 mg/m² + 5-FU 400 mg/m² bolus + 2400 mg/m² CI) q2w × 12 cycles adjuvant",
            "second_line":      "FOLFIRI + bevacizumab (at recurrence, Stage IV intent)",
            "nccn_category":    "1",
            "contraindications": ["Severe peripheral neuropathy ≥Grade 2 (oxaliplatin dose reduction)", "DPD deficiency (5-FU toxicity — DPYD genotype testing)", "Active cardiac arrhythmia (5-FU CI risk)"],
            "monitoring":       ["CBC + CMP q2 weeks (each cycle)", "Peripheral neuropathy grading q2 weeks", "CEA q3-6 months × 5 years", "Colonoscopy 1 year post-surgery"],
        }
    ),
    (
        "Colon adenocarcinoma. pT4a N1b M1a (liver only, resectable). KRAS/NRAS/BRAF wild-type. MSS.",
        {
            "first_line":       "FOLFOX or FOLFIRI + cetuximab (NCCN Category 1 — RAS/BRAF WT, left-sided, conversion intent)",
            "second_line":      "Surgical resection of liver metastases if response achieved (R0 intent); then FOLFOX adjuvant post-hepatectomy",
            "nccn_category":    "1",
            "contraindications": ["Panitumumab not preferred over cetuximab for conversion (similar efficacy; cetuximab data more mature for conversion)", "Bevacizumab in perioperative setting — hold 6 weeks pre-surgery"],
            "monitoring":       ["CEA q2 months", "CT CAP q8 weeks (response assessment)", "Skin toxicity (cetuximab acneiform rash — predictor of response)", "LFTs (hepatic mets)"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N2b M1b (lung + liver). BRAF V600E. MSS.",
        {
            "first_line":       "FOLFOXIRI + bevacizumab (BRAF V600E MSS — high proliferative burden, maximum upfront cytotoxic therapy)",
            "second_line":      "Encorafenib 300 mg PO daily + binimetinib 45 mg PO BID + cetuximab q2w (BEACON trial, NCCN Category 1 post-first-line BRAF V600E)",
            "nccn_category":    "1",
            "contraindications": ["FOLFOXIRI in PS ≥2 (high toxicity)", "Anti-EGFR monotherapy (BRAF V600E — likely primary resistance)", "Encorafenib triplet first-line (insufficient evidence)"],
            "monitoring":       ["CBC q2 weeks", "LFTs q3 weeks (FOLFOXIRI hepatotoxicity)", "CEA q2 months", "CT q8 weeks", "Dermatology (encorafenib cuSCC)"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N1a M1a (liver). MSI-H/dMMR. BRAF V600E co-mutation.",
        {
            "first_line":       "Pembrolizumab 200 mg IV q3w (MSI-H/dMMR Stage IV, KEYNOTE-177, NCCN Category 1)",
            "second_line":      "FOLFOXIRI + bevacizumab (IO progression in MSI-H BRAF-mutant — chemotherapy backbone)",
            "nccn_category":    "1",
            "contraindications": ["Anti-EGFR therapy (BRAF V600E mutant — excluded)", "Active autoimmune disease (IO contraindication)"],
            "monitoring":       ["TFTs q6 weeks", "LFTs q3 weeks", "CEA q2 months", "CT q8-12 weeks", "Colitis monitoring (pembrolizumab diarrhoea)"],
        }
    ),
    (
        "Colon adenocarcinoma. pT4b N2b M0 (Stage IIIC). MSS. Resected. DPYD *2A heterozygous.",
        {
            "first_line":       "CAPOX × 8 cycles adjuvant (capecitabine 1000 mg/m² BID d1-14 — dose reduce 25% for DPYD *2A; oxaliplatin 130 mg/m² d1 q3w)",
            "second_line":      "FOLFIRI (at recurrence, Stage IV)",
            "nccn_category":    "1",
            "contraindications": ["Full-dose capecitabine (DPYD *2A — severe fluoropyrimidine toxicity risk)", "FOLFOX without DPYD adjustment"],
            "monitoring":       ["DPYD genotype-adjusted dose from cycle 1", "CBC q3 weeks", "Hand-foot syndrome q3 weeks (capecitabine)", "Neuropathy q3 weeks"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N0 M0 (Stage II). MSI-H/dMMR. Lynch syndrome (MLH1 germline). High-risk: T4 absent.",
        {
            "first_line":       "Surveillance (Stage II MSI-H/dMMR — adjuvant chemotherapy may not improve OS in MSI-H Stage II per MOSAIC data)",
            "second_line":      "Pembrolizumab (Stage IV recurrence — MSI-H, Category 1)",
            "nccn_category":    "2A",
            "contraindications": ["FOLFOX adjuvant if MSI-H Stage II low-risk (NCCN 2A recommendation — potential lack of benefit)", "Adjuvant IO not standard for Stage II"],
            "monitoring":       ["CEA q3-6 months", "Colonoscopy 1 year", "Lynch syndrome surveillance protocol (annual colonoscopy, gynaecological surveillance)", "Family cascade testing"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N1b M0 (Stage IIIB). NTRK2 fusion. Post-FOLFOX × 6 months. Metastatic recurrence.",
        {
            "first_line":       "Larotrectinib 100 mg PO BID (NTRK fusion-positive — tumour-agnostic, NCCN Category 1)",
            "second_line":      "Entrectinib 600 mg PO daily (NTRK, larotrectinib intolerance or resistance)",
            "nccn_category":    "1",
            "contraindications": ["Strong CYP3A4 inhibitors/inducers (larotrectinib metabolised by CYP3A4)", "Moderate-severe hepatic impairment (dose modification required)"],
            "monitoring":       ["LFTs q2 weeks × 6 months, then monthly", "Neurotoxicity (dizziness, gait)", "CT q8-12 weeks"],
        }
    ),
    (
        "Colon adenocarcinoma. pT4a N2a M0 (Stage IIIC). HER2 amplified (IHC 3+). KRAS/NRAS/BRAF wild-type. MSS. Post-FOLFOX adjuvant. Recurrence at 8 months.",
        {
            "first_line":       "FOLFIRI + bevacizumab (first-line metastatic, standard backbone even in HER2+)",
            "second_line":      "Trastuzumab 8 mg/kg loading then 6 mg/kg q3w + tucatinib 300 mg PO BID (HER2+, post-first-line, MOUNTAINEER trial, NCCN Category 2A)",
            "nccn_category":    "2A",
            "contraindications": ["Trastuzumab + lapatinib first-line (insufficient evidence vs MOUNTAINEER)", "Anti-EGFR with bevacizumab (contraindicated — KRAS WT but anti-EGFR requires bevacizumab washout)"],
            "monitoring":       ["LVEF at baseline q3 months (trastuzumab cardiac toxicity)", "LFTs q3 weeks (tucatinib hepatotoxicity)", "Diarrhoea management protocol (tucatinib)", "CEA q2 months"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N0 M1c (peritoneal carcinomatosis, non-resectable). KRAS G12C. MSS.",
        {
            "first_line":       "FOLFOX + bevacizumab (first-line — no anti-EGFR due to KRAS mutation)",
            "second_line":      "Adagrasib 400 mg PO BID (KRAS G12C CRC, KRYSTAL-1 trial, NCCN 2A) after platinum failure",
            "nccn_category":    "2A",
            "contraindications": ["Cetuximab or panitumumab (KRAS mutant — Category 1 contraindication)", "Bevacizumab hold ≥6 weeks pre-surgery if cytoreductive surgery considered"],
            "monitoring":       ["CEA q2 months", "CT q8 weeks (peritoneal response challenging to assess — PET/CT adjunct)", "VEGF-related toxicities: HTN, proteinuria q3 weeks"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV. POLE exon 9 mutation. Ultra-high TMB (120 mut/Mb). MSS on PCR.",
        {
            "first_line":       "Pembrolizumab 200 mg IV q3w (POLE-mutated/TMB-high — IO response regardless of MSI, NCCN Category 2A)",
            "second_line":      "Nivolumab + ipilimumab (POLE/TMB-high — alternative or progression)",
            "nccn_category":    "2A",
            "contraindications": ["FOLFOX alone (likely inferior — POLE tumours are highly immunogenic)", "Active severe autoimmune disease"],
            "monitoring":       ["TFTs, LFTs q3 weeks", "Auto-immune screen at baseline", "CT q8 weeks response assessment"],
        }
    ),
    (
        "Colon adenocarcinoma. pT2 N0 M0 (Stage I). Complete resection.",
        {
            "first_line":       "Surveillance — no adjuvant chemotherapy for Stage I colon cancer",
            "second_line":      "FOLFOX × 6 months if Stage I revised to IIA-IIB on final pathology review",
            "nccn_category":    "1",
            "contraindications": ["Adjuvant chemotherapy (Stage I — no evidence of benefit, potential harm)"],
            "monitoring":       ["CEA q3-6 months × 3 years", "CT CAP q12 months × 3 years", "Colonoscopy 1 year post-resection"],
        }
    ),
    (
        "Rectal adenocarcinoma. cT3 N1 M0 (Stage IIIB). MSS. KRAS G13D.",
        {
            "first_line":       "Total neoadjuvant therapy: FOLFOX × 4 months then long-course CRT (45 Gy + capecitabine) then TME surgery",
            "second_line":      "Adjuvant FOLFOX × 4 cycles post-TME if pT3-4 or pN+",
            "nccn_category":    "1",
            "contraindications": ["Short-course RT then immediate surgery (appropriate if technically resectable, but TNT preferred for organ preservation intent)", "Anti-EGFR agents (KRAS mutant)"],
            "monitoring":       ["MRI pelvis pre/post induction chemo and post-CRT", "CEA q3 months", "Lower GI toxicity monitoring during CRT"],
        }
    ),
    (
        "Colon adenocarcinoma. pT3 N0 M0 (Stage IIB — T4a equivalent invading peritoneum). MSS. KRAS mutant. High-risk.",
        {
            "first_line":       "FOLFOX × 6 months adjuvant (Stage IIB high-risk, NCCN Category 2B for KRAS mutant — individual risk-benefit discussion)",
            "second_line":      "FOLFIRI + bevacizumab (at metastatic recurrence)",
            "nccn_category":    "2B",
            "contraindications": ["DPYD testing before 5-FU initiation", "Anti-EGFR therapy (KRAS mutant)"],
            "monitoring":       ["CBC + CMP q2 weeks", "Peripheral neuropathy assessment", "CEA q3-6 months × 5 years"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV (liver + lung). KRAS/NRAS/BRAF wild-type. MSS. Right-sided primary.",
        {
            "first_line":       "FOLFOX + bevacizumab (right-sided primary — anti-EGFR less effective regardless of RAS status per CALGB/SWOG 80405)",
            "second_line":      "FOLFIRI + bevacizumab (continue VEGF inhibition beyond progression, TML/ML18147 data)",
            "nccn_category":    "1",
            "contraindications": ["Cetuximab/panitumumab as first-line for right-sided CRC (inferior OS vs bevacizumab regardless of RAS WT status)"],
            "monitoring":       ["CT q8 weeks", "CEA q2 months", "Bevacizumab toxicities: BP weekly, proteinuria q3 weeks, wound healing monitoring"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV. Fifth-line (post-FOLFOX, FOLFIRI, bevacizumab, regorafenib, TAS-102). Adequate PS.",
        {
            "first_line":       "Fruquintinib 5 mg PO daily × 21 days q4w (FRESCO-2 trial — later-line after regorafenib/TAS-102)",
            "second_line":      "Clinical trial enrollment (preferred in heavily pre-treated MSS CRC)",
            "nccn_category":    "2A",
            "contraindications": ["Active haemorrhage (fruquintinib — VEGFR inhibitor)", "Uncontrolled hypertension", "ECOG PS ≥3"],
            "monitoring":       ["Blood pressure daily × 1 week, then weekly", "LFTs q3 weeks", "CT q8 weeks"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV. Post-FOLFOX failure. MSS. KRAS WT (left-sided primary).",
        {
            "first_line":       "FOLFIRI + panitumumab (NCCN Category 1 — RAS WT left-sided second-line; ASPECCT showed panitumumab non-inferior to cetuximab)",
            "second_line":      "TAS-102 (trifluridine + tipiracil) 35 mg/m² PO BID d1-5, 8-12 q4w (post-anti-EGFR failure)",
            "nccn_category":    "1",
            "contraindications": ["Severe hypomagnesaemia (panitumumab — electrolyte replacement required)", "Pre-existing severe skin conditions (acneiform rash)", "Right-sided primary (anti-EGFR benefit diminished)"],
            "monitoring":       ["Skin toxicity assessment (panitumumab rash — graded q3 weeks)", "Electrolytes weekly (hypomagnesaemia)", "CBC q2 weeks (FOLFIRI)"],
        }
    ),
    (
        "Colon adenocarcinoma. pT4a N0 M0 (Stage IIB). MSI-H. BRAF wild-type. Lynch syndrome confirmed.",
        {
            "first_line":       "Observation vs FOLFOX discussion (Stage IIB high-risk MSI-H — some data suggest lack of benefit; MDT decision)",
            "second_line":      "Pembrolizumab (Stage IV recurrence)",
            "nccn_category":    "2A",
            "contraindications": ["5-FU monotherapy (MOSAIC: no benefit MSI-H Stage II from adjuvant chemo, possible harm)", "IO adjuvant (not standard Stage II)"],
            "monitoring":       ["Lynch surveillance protocol: colonoscopy q1-2 years", "Gynaecological/urological surveillance", "CEA q6 months"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV. RAS/BRAF wild-type. MSS. Lung-only oligo-metastatic (2 lesions). Resectable.",
        {
            "first_line":       "Perioperative FOLFOX × 3 cycles → pulmonary metastatectomy (R0 intent) → FOLFOX × 3 cycles post-surgery (NCCN Category 2A)",
            "second_line":      "FOLFIRI + cetuximab (if metastatic progression after metastatectomy)",
            "nccn_category":    "2A",
            "contraindications": ["Bevacizumab within 6 weeks of surgery (wound dehiscence)", "Cemiplimab/IO — no indication in MSS resectable"],
            "monitoring":       ["CT CAP q8 weeks perioperative", "Post-resection CEA q3 months × 2 years", "Neuropathy assessment (oxaliplatin)"],
        }
    ),
    (
        "Colon adenocarcinoma. Stage IV (liver only). MSI-H. Treatment-naive. Fit patient.",
        {
            "first_line":       "Pembrolizumab 200 mg IV q3w (MSI-H/dMMR Stage IV, KEYNOTE-177, NCCN Category 1 preferred over chemotherapy first-line)",
            "second_line":      "Nivolumab + ipilimumab (MSI-H, CHECKMATE 142 — Category 1 alternative first-line)",
            "nccn_category":    "1",
            "contraindications": ["FOLFOX first-line (inferior PFS vs pembrolizumab in MSI-H CRC, KEYNOTE-177)", "Active autoimmune disease requiring systemic immunosuppression"],
            "monitoring":       ["TFTs q6 weeks", "LFTs q3 weeks", "Glucose (immune endocrinopathy)", "CT q8-12 weeks", "Lynch syndrome germline confirmation"],
        }
    ),

    # ── Benign (8 examples) ───────────────────────────────────────────────────
    (
        "Hyperplastic polyp of the colon, no dysplasia.",
        {
            "first_line":       "Surveillance colonoscopy in 10 years (low-risk hyperplastic polyp, no treatment required)",
            "second_line":      "No systemic therapy",
            "nccn_category":    "1",
            "contraindications": ["Systemic chemotherapy — no indication"],
            "monitoring":       ["Repeat colonoscopy per ACG surveillance interval", "Lifestyle: diet, smoking, BMI"],
        }
    ),
    (
        "Tubular adenoma with low-grade dysplasia, completely excised, no invasion.",
        {
            "first_line":       "Colonoscopy in 3-5 years (standard post-adenoma surveillance)",
            "second_line":      "No systemic therapy",
            "nccn_category":    "1",
            "contraindications": ["Adjuvant chemotherapy — not indicated for excised adenoma"],
            "monitoring":       ["Colonoscopy per ACG guidelines (3 years if ≥3 adenomas or advanced histology; 5-10 years if 1-2 low-grade adenomas)"],
        }
    ),
    (
        "Lung biopsy: organising pneumonia. No malignancy.",
        {
            "first_line":       "Prednisolone 40-60 mg PO daily for 4-6 weeks (organising pneumonia standard treatment) — managed by pulmonology",
            "second_line":      "Azathioprine + prednisolone (steroid-dependent organising pneumonia)",
            "nccn_category":    "1",
            "contraindications": ["Chemotherapy — not indicated for non-neoplastic OP"],
            "monitoring":       ["HRCT chest at 6-8 weeks (treatment response)", "Spirometry before/after corticosteroids", "Blood glucose (steroid-induced hyperglycaemia)"],
        }
    ),
    (
        "Hamartoma of the lung, 1.4 cm, completely resected.",
        {
            "first_line":       "Observation — complete resection of pulmonary hamartoma is curative",
            "second_line":      "No systemic therapy required",
            "nccn_category":    "1",
            "contraindications": ["Adjuvant chemotherapy — not indicated for benign hamartoma"],
            "monitoring":       ["CT chest at 12 months post-resection to confirm no recurrence", "Annual CT if size warranted concerns"],
        }
    ),
    (
        "Tubulovillous adenoma, focal high-grade dysplasia, no submucosal invasion, excised.",
        {
            "first_line":       "Colonoscopy in 3-6 months to confirm complete excision, then 3-year interval",
            "second_line":      "Surgical resection if endoscopic clearance unconfirmed",
            "nccn_category":    "1",
            "contraindications": ["Adjuvant chemotherapy — no indication for HGD without invasion"],
            "monitoring":       ["Confirmatory colonoscopy q3-6 months × 1, then annual if prior HGD", "CEA only if progression to invasive carcinoma"],
        }
    ),
    (
        "Atypical adenomatous hyperplasia (AAH) of the lung, 0.3 cm. Pre-invasive.",
        {
            "first_line":       "CT surveillance per Fleischner Society guidelines (sub-solid nodule <6 mm — no follow-up required in low-risk patients)",
            "second_line":      "Annual CT if patient has high-risk features (heavy smoker, family history)",
            "nccn_category":    "1",
            "contraindications": ["Surgical resection for <6 mm AAH — not indicated unless high-risk composite", "Systemic therapy — not indicated"],
            "monitoring":       ["Low-dose CT chest per Fleischner Society", "Smoking cessation counselling"],
        }
    ),
    (
        "Carcinoid tumorlet of the lung, 0.2 cm, Ki-67 <2%, nodes negative.",
        {
            "first_line":       "Observation — carcinoid tumorlet (<5 mm) is incidental finding, no treatment required",
            "second_line":      "If typical carcinoid grows >5 mm: surgical resection consideration",
            "nccn_category":    "1",
            "contraindications": ["Octreotide/somatostatin analogues — not indicated for tumorlet (non-functional, sub-5 mm)", "Everolimus — not indicated"],
            "monitoring":       ["CT chest annually", "Serum CgA only if symptomatic or growing lesion", "24h urine 5-HIAA if functional symptoms"],
        }
    ),
    (
        "Serrated adenoma of the colon, no dysplasia, completely excised.",
        {
            "first_line":       "Colonoscopy in 3-5 years (sessile serrated adenoma without dysplasia, <10 mm — standard surveillance)",
            "second_line":      "No systemic therapy",
            "nccn_category":    "1",
            "contraindications": ["Adjuvant chemotherapy — not indicated for benign serrated adenoma"],
            "monitoring":       ["Colonoscopy 3-year interval (ACG: SSA ≥10 mm or with dysplasia → 3 years; <10 mm without dysplasia → 5 years)"],
        }
    ),
]

assert len(_TREATMENT_EXAMPLES) == 50, f"Expected 50, got {len(_TREATMENT_EXAMPLES)}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B for treatment plan generation on AMD MI300X."
    )
    p.add_argument("--base_model",  default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir",  default="aob/ml/models/checkpoints/treatment_lora")
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--max_steps",   type=int,   default=-1)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_r",      type=int,   default=8)
    p.add_argument("--lora_alpha",  type=int,   default=16)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--max_seq_len", type=int,   default=768)
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.training.lora_trainer import LoRATrainingSpec, train_lora_adapter

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("=" * 60)
    log.info("  AOB Treatment Specialist — LoRA Fine-Tune (Track 2)")
    log.info("  AMD MI300X · ROCm · Optimum-AMD")
    log.info("=" * 60)

    spec = LoRATrainingSpec(
        task_name="treatment",
        prompt_template=PROMPT_TEMPLATE,
        examples=_TREATMENT_EXAMPLES,
        output_schema_keys=OUTPUT_SCHEMA_KEYS,
        output_dir=Path(args.output_dir),
    )

    report = train_lora_adapter(
        spec,
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
    )

    log.info("=" * 60)
    log.info("  Training Summary — Treatment Specialist")
    log.info("=" * 60)
    log.info(f"  Train loss        : {report.training['train_loss']}")
    log.info(f"  Eval exact-match  : {report.eval['exact_match']:.1%}")
    log.info(f"  Schema compliance : {report.eval['schema_compliance']:.1%}")
    log.info(f"  Adapter saved to  : {report.adapter_path}")
    log.info(f"  Optimum-AMD       : {'applied' if report.optimum_applied else 'skipped'}")
    log.info("=" * 60)
    log.info("\nTo serve with all specialists:")
    log.info("  bash scripts/serve_specialists.sh")


if __name__ == "__main__":
    main()
