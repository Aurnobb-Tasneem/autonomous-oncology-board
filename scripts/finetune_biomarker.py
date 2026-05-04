"""
scripts/finetune_biomarker.py
==============================
LoRA fine-tuning of Llama-3.1-8B-Instruct for oncology biomarker panel
extraction on AMD MI300X.

Track 2 (Fine-Tuning on AMD GPUs) deliverable — Biomarker Specialist.

What this does
--------------
1. Creates 50 examples mapping pathology text → structured biomarker JSON.
2. Trains LoRA on meta-llama/Llama-3.1-8B-Instruct via the generic lora_trainer.
3. Integrates Optimum-AMD when available.
4. Saves the adapter to <output_dir>/ with training_report.json.
5. The adapter is served alongside tnm_specialist via vLLM --lora-modules
   (see scripts/serve_specialists.sh).

VRAM budget on AMD MI300X:
    Llama-3.1-8B (bf16)    ~ 16 GB
    LoRA trainable params   <  1 GB
    Activations + grad      ~  8 GB
    Total                   ~ 25 GB  (shares pool with GigaPath + 70B)

Usage
-----
    export HF_TOKEN=hf_...
    python scripts/finetune_biomarker.py

    # Quick smoke (50 steps):
    python scripts/finetune_biomarker.py --max_steps 50

Output JSON schema:
    {"tests_required": [...], "gated_therapies": [...], "rationale": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("finetune_biomarker")

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified molecular oncologist.
Given a pathology description, output ONLY a JSON object with keys:
  tests_required (list of molecular/IHC tests),
  gated_therapies (list of targeted/immunotherapy options pending test results),
  rationale (one sentence citing NCCN guideline rationale).
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

OUTPUT_SCHEMA_KEYS = {"tests_required", "gated_therapies", "rationale"}

# ── 50-example training dataset ───────────────────────────────────────────────
# Covers: NSCLC (adenocarcinoma, squamous), colon, lung benign, colon benign.
# Biomarkers: EGFR, KRAS, ALK, ROS1, BRAF, NTRK, PD-L1, MSI-H, HER2, MET.

_BIOMARKER_EXAMPLES: list[tuple[str, dict]] = [
    # ── Lung Adenocarcinoma (20 examples) ────────────────────────────────────
    (
        "Lung adenocarcinoma, 3.2 cm, peripheral, glandular pattern. Stage IIB.",
        {
            "tests_required": ["EGFR mutation panel (exons 18-21)", "ALK FISH/IHC", "ROS1 FISH/IHC", "PD-L1 IHC (22C3)", "KRAS G12C assay", "NTRK fusion panel"],
            "gated_therapies": ["osimertinib (EGFR exon 19/21+)", "alectinib (ALK+)", "entrectinib (ROS1/NTRK+)", "pembrolizumab (PD-L1 ≥50%)", "sotorasib (KRAS G12C+)"],
            "rationale": "NCCN NSCLC v3.2025 Category 1: comprehensive molecular profiling required before first-line therapy selection in metastatic/locally advanced adenocarcinoma.",
        }
    ),
    (
        "EGFR exon 19 deletion confirmed lung adenocarcinoma, Stage IV. Brain metastasis.",
        {
            "tests_required": ["T790M resistance testing on progression", "MET amplification", "HER2 amplification", "BRAF V600E"],
            "gated_therapies": ["osimertinib 80 mg PO daily (NCCN Category 1)", "CNS-penetrant dosing review", "osimertinib 160 mg if T790M+ on first-gen TKI"],
            "rationale": "NCCN NSCLC v3.2025: osimertinib first-line for EGFR exon 19del; CNS metastasis does not change first-line choice; T790M testing reserved for progression.",
        }
    ),
    (
        "ALK-rearranged lung adenocarcinoma, Stage IIIA. No prior systemic therapy.",
        {
            "tests_required": ["ALK IHC D5F3 confirmatory", "PD-L1 IHC", "EGFR co-mutation screen"],
            "gated_therapies": ["alectinib 600 mg PO BID (NCCN Category 1)", "brigatinib (ALK+ alternative)"],
            "rationale": "NCCN NSCLC: alectinib is preferred first-line for ALK-positive NSCLC; superior CNS penetration vs crizotinib.",
        }
    ),
    (
        "Lung adenocarcinoma, KRAS G12C mutation, Stage IV. PD-L1 TPS 25%.",
        {
            "tests_required": ["STK11/KEAP1 co-mutation panel", "TMB-high assessment"],
            "gated_therapies": ["sotorasib 960 mg PO daily (KRAS G12C+, post-platinum)", "adagrasib (KRAS G12C+, post-platinum)", "pembrolizumab + platinum (first-line if PD-L1 <50%)"],
            "rationale": "NCCN NSCLC: KRAS G12C inhibitors are second-line post-platinum/IO; STK11 co-mutation predicts reduced IO efficacy.",
        }
    ),
    (
        "ROS1 fusion-positive lung adenocarcinoma, Stage IV. No brain metastasis.",
        {
            "tests_required": ["ROS1 IHC/FISH confirmation", "PD-L1 IHC", "Next-gen sequencing for fusion partner"],
            "gated_therapies": ["entrectinib (ROS1+, CNS activity)", "crizotinib (ROS1+)", "lorlatinib (post-crizotinib resistance)"],
            "rationale": "NCCN NSCLC: entrectinib preferred for ROS1+ due to CNS penetrance; crizotinib acceptable alternative.",
        }
    ),
    (
        "MET exon 14 skipping mutation, lung adenocarcinoma, Stage IV.",
        {
            "tests_required": ["MET IHC", "MET copy number (FISH)", "co-mutation screen (KRAS, EGFR)"],
            "gated_therapies": ["tepotinib (MET exon 14+)", "capmatinib (MET exon 14+)", "savolitinib (MET exon 14+)"],
            "rationale": "NCCN NSCLC: MET exon 14 skipping is a Category 1 actionable alteration; MET inhibitor monotherapy recommended.",
        }
    ),
    (
        "BRAF V600E-mutant lung adenocarcinoma, Stage IIIB. No prior targeted therapy.",
        {
            "tests_required": ["BRAF V600E confirmatory sequencing", "MEK co-mutation screen", "PD-L1 IHC"],
            "gated_therapies": ["dabrafenib + trametinib (BRAF V600E+, Category 2A)", "vemurafenib + cobimetinib"],
            "rationale": "NCCN NSCLC: BRAF V600E doublet targeted therapy (BRAF+MEK inhibition) preferred over monotherapy.",
        }
    ),
    (
        "NTRK3 fusion-positive lung adenocarcinoma, Stage IV. Young patient, no prior therapy.",
        {
            "tests_required": ["NTRK1/2/3 NGS panel", "NTRK IHC (as screen)", "PD-L1 IHC"],
            "gated_therapies": ["larotrectinib (NTRK fusion+)", "entrectinib (NTRK fusion+)", "repotrectinib (post-larotrectinib resistance)"],
            "rationale": "NCCN NSCLC: TRK inhibitors are tumour-agnostic first-line for NTRK fusion-positive cancers; larotrectinib 100 mg BID.",
        }
    ),
    (
        "Lung adenocarcinoma, PD-L1 TPS 85%, no driver mutation identified. Stage IV.",
        {
            "tests_required": ["TMB-high assessment (≥10 mut/Mb)", "MSI-H testing", "STK11/KEAP1 mutation screen"],
            "gated_therapies": ["pembrolizumab monotherapy (PD-L1 ≥50%, NCCN Category 1)", "nivolumab + ipilimumab (if high TMB)"],
            "rationale": "NCCN NSCLC: pembrolizumab monotherapy first-line for PD-L1 ≥50% without driver mutations; STK11 co-mutation may blunt IO response.",
        }
    ),
    (
        "HER2-amplified lung adenocarcinoma (IHC 3+), Stage IV.",
        {
            "tests_required": ["HER2 FISH confirmatory", "HER2 exon 20 insertion NGS", "EGFR/ALK/ROS1 exclusion panel"],
            "gated_therapies": ["trastuzumab deruxtecan (HER2 3+ or exon 20 ins)", "poziotinib (HER2 exon 20 ins)", "mobocertinib (EGFR/HER2 exon 20 ins)"],
            "rationale": "NCCN NSCLC 2025: trastuzumab deruxtecan (T-DXd) is Category 2A for HER2-overexpressing/amplified NSCLC.",
        }
    ),
    (
        "Well-differentiated lung adenocarcinoma, 1.1 cm, Stage IA1. No nodal involvement.",
        {
            "tests_required": ["EGFR mutation panel", "ALK IHC", "ROS1 FISH", "PD-L1 IHC"],
            "gated_therapies": ["osimertinib adjuvant (EGFR+, post-resection Stage IB-IIIA)", "alectinib adjuvant (ALK+, if resected)"],
            "rationale": "NCCN NSCLC: molecular profiling recommended for resected adenocarcinoma to guide adjuvant targeted therapy eligibility.",
        }
    ),
    (
        "Lung adenocarcinoma, EGFR L858R, prior osimertinib. Progressive disease at 14 months.",
        {
            "tests_required": ["Liquid biopsy ctDNA — EGFR C797S, amplification", "MET amplification FISH", "SCLC transformation biopsy"],
            "gated_therapies": ["platinum + pemetrexed + bevacizumab (post-osimertinib)", "amivantamab (EGFR/MET bispecific, post-osimertinib)", "clinical trial enrollment"],
            "rationale": "NCCN NSCLC: after osimertinib failure, platinum doublet remains standard; resistance mechanism guides novel therapy selection.",
        }
    ),
    (
        "Micropapillary-predominant lung adenocarcinoma, 4.5 cm, Stage IIB.",
        {
            "tests_required": ["EGFR exon 18-21 panel", "ALK IHC/FISH", "ROS1 IHC/FISH", "KRAS G12C", "PD-L1 IHC", "Broad NGS panel"],
            "gated_therapies": ["osimertinib adjuvant (EGFR+)", "atezolizumab adjuvant (PD-L1+ post-resection)", "cisplatin + pemetrexed adjuvant (no driver)"],
            "rationale": "NCCN NSCLC: micropapillary pattern confers high recurrence risk; comprehensive profiling mandatory for adjuvant therapy optimisation.",
        }
    ),
    (
        "Lung adenocarcinoma in situ (AIS), 0.7 cm, bronchoscopic biopsy.",
        {
            "tests_required": ["EGFR mutation panel (optional — preinvasive)", "Ki-67 proliferation index"],
            "gated_therapies": [],
            "rationale": "NCCN NSCLC: AIS is preinvasive; molecular profiling is optional and does not alter management (surveillance/resection is standard).",
        }
    ),
    (
        "Lung adenocarcinoma, STK11 mutation, KRAS G12C co-mutation. Stage IV.",
        {
            "tests_required": ["PD-L1 IHC (likely negative/low in STK11-mutant)", "TMB assessment", "LKB1 protein IHC"],
            "gated_therapies": ["sotorasib + carboplatin + pemetrexed (investigational)", "adagrasib (KRAS G12C monotherapy, second-line)"],
            "rationale": "NCCN NSCLC: STK11/LKB1 mutation predicts IO resistance; KRAS G12C inhibitor + chemotherapy combinations under investigation.",
        }
    ),
    (
        "Lung adenocarcinoma, high TMB (22 mut/Mb), PD-L1 TPS 15%. Stage IV.",
        {
            "tests_required": ["MSI-H confirmation (PCR)", "dMMR IHC", "POLE mutation screen"],
            "gated_therapies": ["nivolumab + ipilimumab (high TMB, NCCN 2A)", "pembrolizumab + platinum + pemetrexed (if PD-L1 <50%)"],
            "rationale": "NCCN NSCLC: high TMB (≥10 mut/Mb) predicts nivolumab/ipilimumab benefit; PD-L1 TPS 15% still warrants IO combination.",
        }
    ),
    (
        "Lung adenocarcinoma, EGFR exon 20 insertion, Stage IV.",
        {
            "tests_required": ["EGFR exon 20 ins specific NGS", "HER2 exon 20 exclusion", "PD-L1 IHC"],
            "gated_therapies": ["amivantamab-vmjw (EGFR exon 20 ins, NCCN 2A)", "mobocertinib (EGFR exon 20 ins)", "platinum + pemetrexed (first-line if no approved agent access)"],
            "rationale": "NCCN NSCLC 2025: EGFR exon 20 insertions are distinct from classical EGFR mutations and do not respond to standard TKIs.",
        }
    ),
    (
        "Acinar-predominant lung adenocarcinoma, pT1c N0 M0, curative resection.",
        {
            "tests_required": ["EGFR mutation panel", "ALK IHC", "ROS1 IHC/FISH", "PD-L1 IHC"],
            "gated_therapies": ["osimertinib 80 mg PO daily × 3 years (EGFR+, IA3-IIIA post-resection)", "observe (if driver-negative Stage IA3)"],
            "rationale": "NCCN NSCLC: ADAURA trial — adjuvant osimertinib reduces recurrence in EGFR-mutant resected NSCLC Stage IB-IIIA.",
        }
    ),
    (
        "Lung adenocarcinoma with liver metastasis. EGFR wild-type, ALK/ROS1 negative. PD-L1 TPS 40%.",
        {
            "tests_required": ["KRAS G12C assay", "MET exon 14 skip", "NTRK panel", "BRAF V600E", "HER2", "TMB", "MSI-H"],
            "gated_therapies": ["pembrolizumab + carboplatin + pemetrexed (PD-L1 1-49%, NCCN Cat 1)", "atezolizumab + bevacizumab + carboplatin + paclitaxel (alternative)"],
            "rationale": "NCCN NSCLC: after exclusion of actionable drivers, platinum-IO combination is standard first-line for PD-L1 1-49%.",
        }
    ),
    (
        "Lung adenocarcinoma, EGFR T790M detected on ctDNA after gefitinib failure.",
        {
            "tests_required": ["Tissue rebiopsy to confirm T790M", "MET amplification", "SCLC transformation histology check"],
            "gated_therapies": ["osimertinib 80 mg PO daily (T790M+, post first-gen TKI, NCCN Category 1)"],
            "rationale": "NCCN NSCLC: osimertinib is Category 1 for T790M-positive progression on first/second-generation EGFR TKIs.",
        }
    ),

    # ── Lung Squamous Cell Carcinoma (10 examples) ───────────────────────────
    (
        "Squamous cell carcinoma of the lung, Stage IIIB. Central location, no prior therapy.",
        {
            "tests_required": ["PD-L1 IHC (22C3)", "TMB-high assessment", "FGFR1/2 amplification (FISH)", "PIK3CA mutation", "DDR2 mutation screen"],
            "gated_therapies": ["pembrolizumab + carboplatin + paclitaxel (NCCN Cat 1)", "nivolumab + ipilimumab (high TMB)", "erdafitinib (FGFR2/3 fusion, investigational)"],
            "rationale": "NCCN NSCLC squamous: actionable alterations are less common; PD-L1 and TMB guide IO selection; FGFR alterations emerging.",
        }
    ),
    (
        "Poorly differentiated lung squamous carcinoma, PD-L1 TPS 70%. Stage IV.",
        {
            "tests_required": ["EGFR exclusion (rare in squamous)", "ALK exclusion", "TMB"],
            "gated_therapies": ["pembrolizumab monotherapy (PD-L1 ≥50%, NCCN Category 1)", "cemiplimab monotherapy (PD-L1 ≥50%)"],
            "rationale": "NCCN NSCLC squamous: pembrolizumab monotherapy is preferred first-line for PD-L1 TPS ≥50% squamous cell carcinoma.",
        }
    ),
    (
        "Squamous cell carcinoma, FGFR1 amplified, Stage IV. Post-platinum failure.",
        {
            "tests_required": ["FGFR1 FISH confirmation", "FGFR2/3 fusion panel", "PD-L1 IHC"],
            "gated_therapies": ["erdafitinib (FGFR2/3 alterations)", "infigratinib (FGFR2/3)", "clinical trial for FGFR1-amp squamous"],
            "rationale": "NCCN NSCLC: FGFR1 amplification in squamous is under investigation; approved agents target FGFR2/3; clinical trial preferred.",
        }
    ),
    (
        "Squamous cell lung carcinoma, central, invading carina. Stage IVA. PD-L1 1-49%.",
        {
            "tests_required": ["PD-L1 IHC confirmatory", "TMB", "EGFR (exclude rare squamous EGFR)", "DDR pathway mutations"],
            "gated_therapies": ["carboplatin + paclitaxel + pembrolizumab (PD-L1 1-49%)", "nivolumab + ipilimumab (high TMB)", "docetaxel + ramucirumab (second-line)"],
            "rationale": "NCCN NSCLC squamous: chemo-IO combination is Category 1 for PD-L1 <50%; unresectable Stage IVA requires systemic therapy.",
        }
    ),
    (
        "Squamous cell carcinoma of lung, high TMB (18 mut/Mb). Stage IV. PD-L1 TPS 10%.",
        {
            "tests_required": ["MSI-H confirmation", "dMMR IHC", "TMB confirmatory panel"],
            "gated_therapies": ["nivolumab + ipilimumab (high TMB, Category 2A)", "pembrolizumab + platinum (Category 1 alternative)"],
            "rationale": "NCCN NSCLC: CheckMate 227 — nivolumab/ipilimumab benefit in squamous with high TMB (≥10 mut/Mb).",
        }
    ),
    (
        "Squamous cell lung carcinoma, Stage IB. Margins clear, no adjuvant chemo given. Recurrence at 2y.",
        {
            "tests_required": ["PD-L1 IHC", "TMB", "Rebiopsy for any new molecular alterations"],
            "gated_therapies": ["pembrolizumab or nivolumab (second-line post-platinum)", "docetaxel + nintedanib (second-line)"],
            "rationale": "NCCN NSCLC squamous: second-line IO or docetaxel combinations per prior therapy; rebiopsy guides molecular re-profiling.",
        }
    ),
    (
        "Lung squamous carcinoma, DDR2 mutation identified. Stage IV.",
        {
            "tests_required": ["DDR2 L239R/S768R confirmatory sequencing", "PIK3CA co-mutation", "PD-L1 IHC"],
            "gated_therapies": ["dasatinib (DDR2 mutation, investigational)", "clinical trial — DDR2-targeted therapy"],
            "rationale": "NCCN NSCLC: DDR2 mutations are rare actionable alterations in squamous; dasatinib has clinical activity (investigational).",
        }
    ),
    (
        "Squamous cell lung carcinoma, PIK3CA E545K mutation. Stage IV. Post-platinum.",
        {
            "tests_required": ["PIK3CA confirmatory NGS", "PTEN IHC", "AKT1 co-mutation"],
            "gated_therapies": ["alpelisib (PIK3CA+, investigational for NSCLC)", "everolimus + erlotinib (clinical trial)", "docetaxel (standard second-line)"],
            "rationale": "NCCN NSCLC: PIK3CA mutations are not yet a standard actionable target in squamous; clinical trial is preferred approach.",
        }
    ),
    (
        "Squamous cell lung carcinoma invading pericardium. Stage IVB. PS 2.",
        {
            "tests_required": ["PD-L1 IHC", "TMB"],
            "gated_therapies": ["pembrolizumab (if PD-L1 ≥50% and PS ≥2)", "single-agent carboplatin (if IO not tolerated)", "best supportive care"],
            "rationale": "NCCN NSCLC: PS 2 patients may receive IO monotherapy if PD-L1 ≥50%; combination chemo-IO has higher toxicity at PS 2.",
        }
    ),
    (
        "Superficial squamous carcinoma confined to bronchial mucosa, Stage 0.",
        {
            "tests_required": ["Narrow-band imaging bronchoscopy", "PD-L1 IHC (optional)"],
            "gated_therapies": [],
            "rationale": "NCCN NSCLC: Stage 0 superficial squamous is managed with photodynamic therapy or bronchoscopic resection; systemic therapy not indicated.",
        }
    ),

    # ── Colon Adenocarcinoma (12 examples) ───────────────────────────────────
    (
        "Colon adenocarcinoma, pT3 N2a M0. KRAS wild-type, BRAF wild-type, MSS.",
        {
            "tests_required": ["RAS extended panel (KRAS/NRAS exons 2/3/4)", "BRAF V600E confirmatory", "MSI-H/dMMR status", "HER2 IHC/FISH"],
            "gated_therapies": ["FOLFOX + bevacizumab (first-line Stage III)", "FOLFOX + cetuximab (RAS/BRAF wild-type, left-sided primary)", "FOLFIRI + panitumumab (alternative anti-EGFR)"],
            "rationale": "NCCN Colon v1.2025: anti-EGFR agents (cetuximab/panitumumab) restricted to RAS/BRAF wild-type, left-sided primaries.",
        }
    ),
    (
        "MSI-H colon adenocarcinoma, pT3 N1a M0. Stage IIIA.",
        {
            "tests_required": ["dMMR IHC (MLH1/MSH2/MSH6/PMS2)", "MLH1 promoter methylation", "BRAF V600E co-mutation screen", "Lynch syndrome germline testing"],
            "gated_therapies": ["FOLFOX adjuvant (Stage III, standard)", "pembrolizumab (Stage IV MSI-H only, not adjuvant)", "Lynch syndrome hereditary counselling"],
            "rationale": "NCCN Colon: MSI-H/dMMR guides IO eligibility in Stage IV; Lynch syndrome testing warranted for germline MLH1/MSH2 variants.",
        }
    ),
    (
        "BRAF V600E-mutant colon adenocarcinoma, pT4a N2b M1a (liver). MSS.",
        {
            "tests_required": ["RAS extended panel (KRAS/NRAS)", "PIK3CA co-mutation", "ERBB2/HER2 amplification"],
            "gated_therapies": ["FOLFOXIRI + bevacizumab (first-line BRAF V600E-mutant)", "encorafenib + binimetinib + cetuximab (post-first-line BRAF V600E+, NCCN Cat 1)", "pembrolizumab not indicated (MSS)"],
            "rationale": "NCCN Colon: BRAF V600E-mutant MSS colon has poor prognosis; encorafenib triplet is Category 1 second-line.",
        }
    ),
    (
        "Colon adenocarcinoma, HER2 amplified (IHC 3+), KRAS/NRAS/BRAF wild-type. Stage IV.",
        {
            "tests_required": ["HER2 FISH confirmatory", "RAS extended panel (repeat)", "MSI-H/dMMR"],
            "gated_therapies": ["trastuzumab + tucatinib (HER2+, post-first-line)", "trastuzumab + lapatinib (HER2+, alternative)", "trastuzumab deruxtecan (investigational, HER2+)"],
            "rationale": "NCCN Colon: HER2 amplification in RAS/BRAF wild-type CRC is actionable with anti-HER2 doublets in second-line (MOUNTAINEER trial).",
        }
    ),
    (
        "Colon adenocarcinoma, MSI-H, BRAF V600E co-mutation. Stage IV (liver + peritoneal).",
        {
            "tests_required": ["dMMR IHC confirmatory", "MLH1 methylation (somatic vs germline MSI-H)", "RAS extended panel"],
            "gated_therapies": ["pembrolizumab monotherapy (MSI-H Stage IV, NCCN Cat 1)", "nivolumab + ipilimumab (MSI-H, NCCN Cat 1 alternative)", "KEYNOTE-177 data supports IO first-line"],
            "rationale": "NCCN Colon: pembrolizumab is Category 1 first-line for MSI-H/dMMR metastatic CRC regardless of BRAF status.",
        }
    ),
    (
        "Colon adenocarcinoma pT2 N0 M0 (Stage I). Complete resection. No adjuvant therapy planned.",
        {
            "tests_required": ["MMR IHC (MLH1/MSH2/MSH6/PMS2)", "Lynch syndrome risk stratification", "KRAS/NRAS/BRAF (baseline for potential future recurrence)"],
            "gated_therapies": [],
            "rationale": "NCCN Colon: Stage I CRC — adjuvant chemotherapy not indicated; MMR testing guides Lynch syndrome counselling.",
        }
    ),
    (
        "Colon adenocarcinoma, NTRK2 fusion-positive, Stage IV. Post-FOLFOX.",
        {
            "tests_required": ["NTRK1/2/3 NGS confirmation", "RAS/BRAF panel", "MSI-H"],
            "gated_therapies": ["larotrectinib (NTRK fusion+, tumour-agnostic)", "entrectinib (NTRK fusion+)", "repotrectinib (post-larotrectinib resistance)"],
            "rationale": "NCCN Colon: TRK inhibitors are tumour-agnostic for NTRK fusion-positive CRC; approved regardless of prior chemotherapy lines.",
        }
    ),
    (
        "Mucinous colon adenocarcinoma, pT4b N2b M0 (Stage IIIC). KRAS G12D mutant.",
        {
            "tests_required": ["RAS extended panel (KRAS exon 2/3/4, NRAS)", "BRAF V600E", "MSI-H/dMMR", "HER2 amplification"],
            "gated_therapies": ["FOLFOX or CAPOX adjuvant (Stage III, Category 1)", "bevacizumab addition (investigational adjuvant, not standard)", "clinical trial for KRAS G12D direct inhibitor"],
            "rationale": "NCCN Colon Stage III: FOLFOX/CAPOX adjuvant is Category 1; KRAS G12D direct inhibitors (adagrasib exon) in clinical trials.",
        }
    ),
    (
        "Colon adenocarcinoma with peritoneal seeding, MSS, KRAS G12V. Elevated CEA.",
        {
            "tests_required": ["RAS/BRAF full panel", "HER2 amplification", "TMB assessment"],
            "gated_therapies": ["FOLFOX/FOLFIRI + bevacizumab (first-line)", "FOLFIRI + aflibercept (second-line after bevacizumab)", "regorafenib / TAS-102 (third-line)"],
            "rationale": "NCCN Colon: peritoneal disease (Stage IVC) — chemotherapy backbone + VEGF inhibitor first-line; anti-EGFR excluded (KRAS mutant).",
        }
    ),
    (
        "Right-sided colon adenocarcinoma, pT3 N1b M0. KRAS G12C mutant.",
        {
            "tests_required": ["KRAS G12C confirmatory NGS", "MSI-H", "BRAF V600E exclusion"],
            "gated_therapies": ["FOLFOX adjuvant (Stage IIIB, standard)", "adagrasib (KRAS G12C, investigational for CRC)", "sotorasib (KRAS G12C, CodeBreak 300 trial)"],
            "rationale": "NCCN Colon: KRAS G12C CRC is under investigation; CodeBreak 300 supports adagrasib in third-line; FOLFOX adjuvant is standard first intent.",
        }
    ),
    (
        "Colon adenocarcinoma, Lynch syndrome confirmed (MLH1 germline). pT3 N0 M0.",
        {
            "tests_required": ["MMR IHC", "Lynch syndrome germline panel (MLH1/MSH2/MSH6/PMS2/EPCAM)", "family cascade testing"],
            "gated_therapies": ["FOLFOX adjuvant (Stage IIB-III, standard)", "aspirin chemoprevention (post-resection per Lynch evidence)", "pembrolizumab not indicated adjuvant Stage II-III"],
            "rationale": "NCCN Colon: Lynch syndrome requires germline confirmation + family cascade; pembrolizumab adjuvant not yet standard for Stage II-III.",
        }
    ),
    (
        "Colon adenocarcinoma, POLE exon 9 mutation, ultra-high TMB (>100 mut/Mb). Stage IV.",
        {
            "tests_required": ["POLE/POLD1 pathogenicity confirmation", "MSI-H PCR (may be MSS despite POLE)", "dMMR IHC"],
            "gated_therapies": ["pembrolizumab (POLE/TMB-high, NCCN 2A)", "nivolumab + ipilimumab (ultra-high TMB)", "investigational POLE-directed trial"],
            "rationale": "NCCN Colon: POLE ultramutated CRC has exceptional IO response regardless of MSI status; TMB >100 mut/Mb is predictive.",
        }
    ),

    # ── Benign tissue (8 examples) ────────────────────────────────────────────
    (
        "Hyperplastic polyp of the colon, no dysplasia. No invasion.",
        {
            "tests_required": [],
            "gated_therapies": [],
            "rationale": "No malignancy; molecular testing not indicated for hyperplastic polyps. Surveillance interval per ACG colonoscopy guidelines.",
        }
    ),
    (
        "Tubular adenoma with low-grade dysplasia, completely excised. No invasion.",
        {
            "tests_required": ["Lynch syndrome MMR IHC if multiple adenomas or family history"],
            "gated_therapies": [],
            "rationale": "Low-grade adenoma — no therapeutic agents indicated; Lynch syndrome MMR testing recommended if risk factors present (NCCN Genetic/Familial High-Risk).",
        }
    ),
    (
        "Lung biopsy: organising pneumonia. No malignancy identified.",
        {
            "tests_required": [],
            "gated_therapies": [],
            "rationale": "Organising pneumonia is a non-neoplastic process; no oncology biomarker workup required. Manage per pulmonary/rheumatology guidelines.",
        }
    ),
    (
        "Hamartoma of the lung, 1.4 cm, completely resected.",
        {
            "tests_required": [],
            "gated_therapies": [],
            "rationale": "Lung hamartoma is benign; no molecular profiling required. Complete resection is curative. No oncology follow-up indicated.",
        }
    ),
    (
        "Tubulovillous adenoma with focal high-grade dysplasia, no submucosal invasion.",
        {
            "tests_required": ["MMR IHC if Lynch syndrome suspicion", "Repeat colonoscopy in 3-6 months to confirm complete excision"],
            "gated_therapies": [],
            "rationale": "High-grade dysplasia without invasion does not require adjuvant systemic therapy; excision confirmation and surveillance are standard.",
        }
    ),
    (
        "Atypical adenomatous hyperplasia (AAH) of the lung, 0.3 cm. Pre-invasive lesion.",
        {
            "tests_required": ["EGFR mutation panel (optional, research context)", "CT surveillance imaging"],
            "gated_therapies": [],
            "rationale": "AAH is a pre-invasive GGO lesion; no systemic therapy indicated. CT surveillance per Fleischner Society guidelines is standard.",
        }
    ),
    (
        "Carcinoid tumorlet of the lung, 0.2 cm, no angioinvasion, nodes negative.",
        {
            "tests_required": ["Chromogranin A/synaptophysin IHC (confirm neuroendocrine)", "Ki-67 proliferation index"],
            "gated_therapies": [],
            "rationale": "Tumorlet (<5 mm) is incidental; Ki-67 <2% confirms typical carcinoid behaviour. No adjuvant oncology therapy indicated.",
        }
    ),
    (
        "Serrated adenoma of the colon, no dysplasia, completely excised.",
        {
            "tests_required": ["MMR IHC if sessile serrated adenoma with dysplasia or size ≥10 mm"],
            "gated_therapies": [],
            "rationale": "Sessile serrated adenoma without dysplasia — no systemic therapy. Surveillance per ACS/ACG guidelines (3-year interval for ≥10 mm).",
        }
    ),
]

assert len(_BIOMARKER_EXAMPLES) == 50, f"Expected 50 examples, got {len(_BIOMARKER_EXAMPLES)}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B for oncology biomarker panel extraction on AMD MI300X."
    )
    p.add_argument("--base_model",  default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output_dir",  default="aob/ml/models/checkpoints/biomarker_lora")
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--max_steps",   type=int,   default=-1)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_r",      type=int,   default=8)
    p.add_argument("--lora_alpha",  type=int,   default=16)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--max_seq_len", type=int,   default=512)
    return p.parse_args()


def main():
    args = parse_args()

    # Import here so that module-level import errors are clear
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.training.lora_trainer import LoRATrainingSpec, train_lora_adapter

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log.info("=" * 60)
    log.info("  AOB Biomarker Specialist — LoRA Fine-Tune (Track 2)")
    log.info("  AMD MI300X · ROCm · Optimum-AMD")
    log.info("=" * 60)
    log.info(f"Base model : {args.base_model}")
    log.info(f"Output dir : {args.output_dir}")
    log.info(f"Examples   : {len(_BIOMARKER_EXAMPLES)}")

    spec = LoRATrainingSpec(
        task_name="biomarker",
        prompt_template=PROMPT_TEMPLATE,
        examples=_BIOMARKER_EXAMPLES,
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
    log.info("  Training Summary — Biomarker Specialist")
    log.info("=" * 60)
    log.info(f"  Train loss        : {report.training['train_loss']}")
    log.info(f"  Eval exact-match  : {report.eval['exact_match']:.1%}")
    log.info(f"  Schema compliance : {report.eval['schema_compliance']:.1%}")
    log.info(f"  Adapter saved to  : {report.adapter_path}")
    log.info(f"  Optimum-AMD       : {'applied' if report.optimum_applied else 'skipped'}")
    log.info("=" * 60)
    log.info("\nTo serve (with tnm_specialist and treatment_specialist):")
    log.info("  bash scripts/serve_specialists.sh")


if __name__ == "__main__":
    main()
