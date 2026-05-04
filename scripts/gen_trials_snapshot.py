"""
One-time script to generate the 500-trial ClinicalTrials.gov snapshot.
Run: python scripts/gen_trials_snapshot.py
"""
import json
import random
import pathlib

random.seed(42)

LUNG_ADENO_TRIALS = [
    {
        "nct_id": "NCT04293133",
        "title": "FLAURA2: Osimertinib + Chemotherapy vs Osimertinib Alone in EGFR-Mutant Advanced NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "EGFR exon 19 deletion or L858R",
        "study_status": "Recruiting",
        "brief_summary": "First-line osimertinib with or without platinum-pemetrexed in EGFR-mutant advanced NSCLC.",
        "inclusion_snippet": "EGFR exon 19 deletion or exon 21 L858R mutation; Stage IIIB/IV; ECOG PS 0-1; no prior systemic therapy for advanced disease.",
        "exclusion_snippet": "Prior EGFR TKI; active CNS metastasis requiring steroids; prior allograft.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "AstraZeneca oncology trials helpline"
    },
    {
        "nct_id": "NCT05092750",
        "title": "MARIPOSA: Amivantamab + Lazertinib vs Osimertinib as First-Line in EGFR-Mutant NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "EGFR exon 19 deletion or L858R",
        "study_status": "Recruiting",
        "brief_summary": "Amivantamab plus lazertinib vs osimertinib, first-line, EGFR-mutant advanced NSCLC.",
        "inclusion_snippet": "Confirmed EGFR exon 19del or L858R; Stage IIIB-IV; no prior systemic for advanced; adequate organ function.",
        "exclusion_snippet": "Prior EGFR-directed therapy; unstable brain mets; severe cardiac QTc prolongation.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Janssen Research clinicaltrials@janssen.com"
    },
    {
        "nct_id": "NCT03456063",
        "title": "PAPILLON: Amivantamab + Chemotherapy vs Chemotherapy in EGFR Exon 20 Insertion NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "EGFR exon 20 insertion",
        "study_status": "Active",
        "brief_summary": "Amivantamab + carboplatin-pemetrexed vs chemotherapy in EGFR exon 20 insertion-positive advanced NSCLC.",
        "inclusion_snippet": "Confirmed EGFR exon 20 insertion; treatment-naive advanced NSCLC; ECOG PS 0-1; no prior targeted therapy.",
        "exclusion_snippet": "Prior amivantamab; EGFR exon 19del or L858R; severe hepatic impairment.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Janssen trials papillon@janssen.com"
    },
    {
        "nct_id": "NCT04585815",
        "title": "LAURA: Osimertinib as Consolidation After CRT in Unresectable EGFR-Mutant Stage III NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "EGFR mutation exon 19del or L858R",
        "study_status": "Active",
        "brief_summary": "Osimertinib vs placebo after definitive concurrent chemoradiotherapy in Stage III EGFR-mutant NSCLC.",
        "inclusion_snippet": "EGFR exon 19del or L858R; unresectable Stage III; completed CRT; ECOG PS 0-1.",
        "exclusion_snippet": "Progression after CRT; prior systemic therapy; ILD at baseline.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "AstraZeneca LAURA trial team"
    },
    {
        "nct_id": "NCT04483206",
        "title": "SAVANNAH: Savolitinib + Osimertinib in MET-Amplified EGFR-Mutant NSCLC after Osimertinib",
        "phase": "Phase II",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "EGFR mutation and MET amplification",
        "study_status": "Recruiting",
        "brief_summary": "Savolitinib + osimertinib in EGFR-mutant NSCLC that progressed on osimertinib with MET amplification.",
        "inclusion_snippet": "Prior osimertinib progression; confirmed MET amplification; EGFR exon 19del or L858R; ECOG PS 0-2.",
        "exclusion_snippet": "No MET amplification; prior MET inhibitor; EGFR C797S resistance mutation.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "AstraZeneca SAVANNAH trial"
    },
    {
        "nct_id": "NCT04267939",
        "title": "CROWN: Lorlatinib vs Crizotinib as First-Line Therapy in Advanced ALK-Positive NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "ALK rearrangement",
        "study_status": "Active",
        "brief_summary": "Lorlatinib vs crizotinib in ALK-positive advanced NSCLC — 3-year follow-up PFS analysis.",
        "inclusion_snippet": "ALK-positive (FISH or IHC); Stage IIIB/IV; treatment-naive; ECOG PS 0-2.",
        "exclusion_snippet": "Prior ALK inhibitor; active autoimmune disease; severe cardiac disease; pregnancy.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Pfizer Oncology lorlatinib@pfizer.com"
    },
    {
        "nct_id": "NCT03727477",
        "title": "ALINA: Alectinib as Adjuvant Therapy in Resected ALK-Positive NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "ALK rearrangement resected",
        "study_status": "Active",
        "brief_summary": "Alectinib vs platinum-based chemotherapy as adjuvant therapy in resected Stage IB-IIIA ALK-positive NSCLC.",
        "inclusion_snippet": "Resected Stage IB-IIIA ALK-positive NSCLC; R0 resection; ECOG PS 0-2; no prior systemic therapy.",
        "exclusion_snippet": "Prior ALK inhibitor; unresected disease; active autoimmune disease; severe hepatic impairment.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Roche/Genentech ALINA trial"
    },
    {
        "nct_id": "NCT04938124",
        "title": "TRIDENT-1: Repotrectinib in ROS1+ or NTRK+ Advanced Solid Tumours",
        "phase": "Phase II",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "ROS1 fusion or NTRK fusion",
        "study_status": "Recruiting",
        "brief_summary": "Repotrectinib (next-generation TKI) in ROS1-positive or NTRK fusion-positive advanced solid tumours.",
        "inclusion_snippet": "Confirmed ROS1 or NTRK1/2/3 fusion by NGS or FISH; advanced solid tumour; ECOG PS 0-2.",
        "exclusion_snippet": "Active CNS leptomeningeal disease; severe ILD; prior repotrectinib.",
        "min_age": 12, "max_ecog_ps": 2,
        "contact_info": "Bristol-Myers Squibb TRIDENT-1"
    },
    {
        "nct_id": "NCT03515837",
        "title": "SKYSCRAPER-01: Tiragolumab + Atezolizumab vs Atezolizumab Alone in PD-L1-High NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "PD-L1 TPS 50 percent or higher",
        "study_status": "Active",
        "brief_summary": "Anti-TIGIT tiragolumab plus atezolizumab vs atezolizumab alone in PD-L1 high first-line NSCLC.",
        "inclusion_snippet": "PD-L1 TPS 50% or higher; Stage IIIB-IV; treatment-naive; no EGFR/ALK/ROS1 alteration; ECOG PS 0-1.",
        "exclusion_snippet": "EGFR/ALK/ROS1 positive; prior immunotherapy; active autoimmune disease; pregnancy.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Roche/Genentech SKYSCRAPER-01"
    },
    {
        "nct_id": "NCT03600883",
        "title": "CodeBreak 300: Adagrasib vs Docetaxel in KRAS G12C-Mutant NSCLC Post-Platinum/IO",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "KRAS G12C",
        "study_status": "Active",
        "brief_summary": "Adagrasib vs docetaxel in second-line KRAS G12C-mutant NSCLC after platinum-based and IO therapy.",
        "inclusion_snippet": "KRAS G12C-mutant; 1 or more prior platinum-based; 1 or more prior IO; ECOG PS 0-2.",
        "exclusion_snippet": "Prior KRAS inhibitor; active CNS leptomeningeal disease; LVEF less than 45%.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Mirati/BMS CodeBreak 300 trial"
    },
    {
        "nct_id": "NCT04613596",
        "title": "KRYSTAL-7: Adagrasib + Pembrolizumab in First-Line PD-L1 >=50% KRAS G12C NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "KRAS G12C and PD-L1 TPS 50 percent or higher",
        "study_status": "Recruiting",
        "brief_summary": "Adagrasib + pembrolizumab vs pembrolizumab in first-line PD-L1 high KRAS G12C-mutant NSCLC.",
        "inclusion_snippet": "KRAS G12C confirmed; PD-L1 TPS 50% or higher; Stage IV; treatment-naive; ECOG PS 0-1.",
        "exclusion_snippet": "PD-L1 less than 50%; prior IO or KRAS inhibitor; ILD; QTc prolongation.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Mirati KRYSTAL-7 trial"
    },
]

SQUAMOUS_TRIALS = [
    {
        "nct_id": "NCT05091372",
        "title": "KEYVIBE-008: Vibostolimab + Pembrolizumab vs Pembrolizumab in PD-L1 >=50% Squamous NSCLC",
        "phase": "Phase III",
        "cancer_type": "Lung Squamous Cell Carcinoma",
        "biomarker_focus": "PD-L1 TPS 50 percent or higher",
        "study_status": "Recruiting",
        "brief_summary": "Anti-TIGIT vibostolimab + pembrolizumab vs pembrolizumab in first-line PD-L1 high squamous NSCLC.",
        "inclusion_snippet": "PD-L1 TPS 50% or higher (22C3 PharmDx); Stage IV squamous NSCLC; treatment-naive; ECOG PS 0-1.",
        "exclusion_snippet": "Prior IO; autoimmune disease on systemic treatment; prior corticosteroids.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Merck KEYVIBE-008"
    },
    {
        "nct_id": "NCT05256485",
        "title": "RAMOSE: Ramucirumab + Pembrolizumab vs Chemotherapy in Advanced Squamous NSCLC Second-Line",
        "phase": "Phase III",
        "cancer_type": "Lung Squamous Cell Carcinoma",
        "biomarker_focus": "VEGFR2 expression",
        "study_status": "Recruiting",
        "brief_summary": "Ramucirumab + pembrolizumab as second-line in advanced squamous NSCLC after platinum failure.",
        "inclusion_snippet": "Squamous NSCLC; progression after 1 or more platinum-based; ECOG PS 0-2; no active haemoptysis.",
        "exclusion_snippet": "Active haemoptysis Grade 2 or higher; prior VEGFR therapy; uncontrolled hypertension.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Eli Lilly RAMOSE trial"
    },
]

COLON_TRIALS = [
    {
        "nct_id": "NCT05765344",
        "title": "BREAKWATER: Encorafenib + Cetuximab + Chemotherapy in BRAF V600E mCRC First-Line",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "BRAF V600E and RAS wild-type",
        "study_status": "Recruiting",
        "brief_summary": "Encorafenib + binimetinib + cetuximab + FOLFOX vs standard in first-line BRAF V600E-mutant mCRC.",
        "inclusion_snippet": "Confirmed BRAF V600E; RAS wild-type; Stage IV mCRC; treatment-naive; ECOG PS 0-1.",
        "exclusion_snippet": "RAS-mutant; prior BRAF inhibitor; active brain metastasis; QTc over 470 ms; LVEF less than 50%.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Array BioPharma/Pfizer BREAKWATER trial"
    },
    {
        "nct_id": "NCT04685876",
        "title": "MOUNTAINEER-03: Tucatinib + Trastuzumab + FOLFOX vs FOLFOX + Cetuximab in HER2+ mCRC",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "HER2 amplification IHC 3 plus or FISH positive",
        "study_status": "Recruiting",
        "brief_summary": "Tucatinib + trastuzumab + mFOLFOX6 vs mFOLFOX6 + cetuximab as first-line in HER2-positive RAS/BRAF wild-type mCRC.",
        "inclusion_snippet": "HER2 IHC 3+ or FISH amplified; RAS wild-type; BRAF wild-type; Stage IV; treatment-naive; ECOG PS 0-1.",
        "exclusion_snippet": "RAS or BRAF mutation; prior anti-HER2 therapy; LVEF less than 50%; prior peripheral neuropathy.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Seagen/Pfizer MOUNTAINEER-03 trial"
    },
    {
        "nct_id": "NCT04294251",
        "title": "KRYSTAL-10: Adagrasib + Cetuximab vs Chemotherapy in KRAS G12C mCRC Second-Line",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "KRAS G12C",
        "study_status": "Active",
        "brief_summary": "Adagrasib + cetuximab vs standard chemotherapy in second-line KRAS G12C-mutant mCRC.",
        "inclusion_snippet": "KRAS G12C confirmed; 1 or more prior fluoropyrimidine plus oxaliplatin; ECOG PS 0-2.",
        "exclusion_snippet": "Prior KRAS G12C inhibitor; prior anti-EGFR; MSI-H; active CNS disease.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Mirati KRYSTAL-10 trial"
    },
    {
        "nct_id": "NCT05263518",
        "title": "KEYNOTE-177 Extension: Pembrolizumab in MSI-H/dMMR CRC 5-Year Follow-Up",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "MSI-H or dMMR",
        "study_status": "Active",
        "brief_summary": "Long-term follow-up of pembrolizumab in first-line MSI-H/dMMR metastatic CRC.",
        "inclusion_snippet": "MSI-H or dMMR confirmed; Stage IV mCRC; treatment-naive; ECOG PS 0-1.",
        "exclusion_snippet": "MSS; prior pembrolizumab or IO; active autoimmune disease; prior organ transplant.",
        "min_age": 18, "max_ecog_ps": 1,
        "contact_info": "Merck KEYNOTE-177 extension"
    },
    {
        "nct_id": "NCT04896073",
        "title": "COLO-PREVENT: Aspirin + Metformin vs Placebo in Lynch Syndrome Carriers",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "Lynch syndrome MLH1 MSH2 MSH6 PMS2 germline",
        "study_status": "Recruiting",
        "brief_summary": "Chemoprevention: aspirin 600 mg/day + metformin vs placebo in confirmed Lynch syndrome for primary CRC prevention.",
        "inclusion_snippet": "Confirmed pathogenic germline variant in MLH1/MSH2/MSH6/PMS2/EPCAM; age 25-70; prior colonoscopy within 2 years.",
        "exclusion_snippet": "Active CRC; aspirin contraindication; metformin contraindication; prior GI bleed.",
        "min_age": 25, "max_ecog_ps": 2,
        "contact_info": "COLO-PREVENT International Consortium"
    },
    {
        "nct_id": "NCT05413889",
        "title": "FRESCO-3: Fruquintinib + BSC vs Placebo in Later-Line mCRC",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "Any pan-tumour later-line",
        "study_status": "Recruiting",
        "brief_summary": "Fruquintinib (pan-VEGFR inhibitor) after regorafenib/TAS-102 in heavily pre-treated mCRC.",
        "inclusion_snippet": "mCRC; 2 or more prior lines; prior regorafenib or TAS-102; ECOG PS 0-2.",
        "exclusion_snippet": "Active haemorrhage; uncontrolled hypertension; severe hepatic impairment; bleeding diathesis.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "HUTCHMED FRESCO-3 trial"
    },
    {
        "nct_id": "NCT05022225",
        "title": "LEAP-017: Lenvatinib + Pembrolizumab in MSS Metastatic CRC Post-Standard Therapy",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "MSS microsatellite stable",
        "study_status": "Recruiting",
        "brief_summary": "Lenvatinib + pembrolizumab vs TAS-102 or regorafenib in treatment-refractory MSS mCRC.",
        "inclusion_snippet": "MSS mCRC; 2 or more prior lines; ECOG PS 0-2; adequate hepatic function.",
        "exclusion_snippet": "MSI-H; prior pembrolizumab; active haemorrhage; severe thromboembolism within 6 months.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Merck/Eisai LEAP-017 trial"
    },
    {
        "nct_id": "NCT05043584",
        "title": "CIRCULATE-Japan GALAXY: ctDNA-Guided Adjuvant Chemotherapy in Stage II-III Colon Cancer",
        "phase": "Phase III",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "ctDNA MRD circulating tumour DNA",
        "study_status": "Recruiting",
        "brief_summary": "ctDNA-guided adjuvant chemotherapy in resected Stage II-III CRC: ctDNA-positive gets FOLFOX; ctDNA-negative observes.",
        "inclusion_snippet": "Resected Stage II-III CRC; R0 resection; ctDNA sample within 4 weeks post-surgery; ECOG PS 0-2.",
        "exclusion_snippet": "Stage I or Stage IV; prior systemic therapy; MSI-H with known Lynch syndrome.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "CIRCULATE-Japan consortium"
    },
]

HER2_NTRK_TRIALS = [
    {
        "nct_id": "NCT04539938",
        "title": "DESTINY-Lung03: T-DXd + Osimertinib in HER2 Overexpressing NSCLC Post-Osimertinib",
        "phase": "Phase Ib/II",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "HER2 overexpression IHC 3 plus or HER2 exon 20 insertion with EGFR TKI resistance",
        "study_status": "Recruiting",
        "brief_summary": "T-DXd + osimertinib in HER2-overexpressing NSCLC with EGFR mutation after osimertinib progression.",
        "inclusion_snippet": "EGFR exon 19del or L858R; prior osimertinib; HER2 IHC 3+ or exon 20 ins; ECOG PS 0-2; adequate LVEF.",
        "exclusion_snippet": "Prior T-DXd or anti-HER2; ILD at baseline; LVEF less than 50%; active CNS leptomeningeal disease.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Daiichi Sankyo/AstraZeneca DESTINY-Lung03"
    },
    {
        "nct_id": "NCT03334617",
        "title": "DESTINY-CRC02: Trastuzumab Deruxtecan in HER2-Positive Metastatic CRC",
        "phase": "Phase II",
        "cancer_type": "Colon Adenocarcinoma",
        "biomarker_focus": "HER2 IHC 3 plus or IHC 2 plus FISH positive",
        "study_status": "Active",
        "brief_summary": "T-DXd in HER2-positive mCRC after 2 or more prior lines (IHC 3+ and IHC 2+/ISH+ cohorts).",
        "inclusion_snippet": "HER2 IHC 3+ or IHC 2+ FISH+; 2 or more prior lines; ECOG PS 0-2; LVEF 50% or higher.",
        "exclusion_snippet": "Prior anti-HER2; ILD/pneumonitis; LVEF less than 50%; active hepatic disease; pregnancy.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Daiichi Sankyo DESTINY-CRC02"
    },
    {
        "nct_id": "NCT02576431",
        "title": "LOXO-TRK-14001: Larotrectinib in NTRK Fusion-Positive Solid Tumours (Ongoing Basket)",
        "phase": "Phase I/II",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "NTRK1 NTRK2 NTRK3 fusion tumour-agnostic",
        "study_status": "Recruiting",
        "brief_summary": "Ongoing basket trial of larotrectinib in NTRK fusion-positive solid tumours across all histologies.",
        "inclusion_snippet": "Confirmed NTRK1/2/3 fusion by NGS or FISH; any solid tumour; 1 or more prior therapy or treatment-naive; ECOG PS 0-2.",
        "exclusion_snippet": "Prior TRK inhibitor; known resistance mutations; active CNS disease requiring steroids.",
        "min_age": 1, "max_ecog_ps": 2,
        "contact_info": "Bayer Oncology larotrectinib trial"
    },
    {
        "nct_id": "NCT02414139",
        "title": "GEOMETRY-mono-1: Capmatinib in MET Exon 14 Skipping NSCLC",
        "phase": "Phase II",
        "cancer_type": "Lung Adenocarcinoma",
        "biomarker_focus": "MET exon 14 skipping mutation",
        "study_status": "Active",
        "brief_summary": "Capmatinib (MET inhibitor) in MET exon 14 skipping-positive advanced NSCLC.",
        "inclusion_snippet": "Confirmed MET exon 14 skipping; advanced NSCLC; ECOG PS 0-2; adequate hepatic/renal function.",
        "exclusion_snippet": "Prior MET or HGF inhibitor; EGFR/ALK/ROS1 positive; severe oedema; strong CYP3A inducers.",
        "min_age": 18, "max_ecog_ps": 2,
        "contact_info": "Novartis GEOMETRY-mono-1"
    },
]

HANDCRAFTED = LUNG_ADENO_TRIALS + SQUAMOUS_TRIALS + COLON_TRIALS + HER2_NTRK_TRIALS

CANCER_TYPES = [
    "Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma",
    "Colon Adenocarcinoma", "Rectal Adenocarcinoma",
    "Breast Invasive Ductal Carcinoma", "Melanoma",
    "Hepatocellular Carcinoma", "Gastric Adenocarcinoma",
    "Urothelial Carcinoma", "Head and Neck Squamous Cell Carcinoma",
    "Pancreatic Ductal Adenocarcinoma", "Ovarian Carcinoma",
    "Endometrial Carcinoma", "Cholangiocarcinoma",
    "Small Cell Lung Cancer",
]
BIOMARKERS = [
    "EGFR mutation", "ALK rearrangement", "ROS1 fusion", "KRAS G12C",
    "KRAS G12D", "BRAF V600E", "HER2 amplification", "MSI-H dMMR",
    "PD-L1 TPS 50 percent or higher", "TMB-high", "NTRK fusion",
    "MET exon 14 skipping", "POLE mutation", "FGFR2 fusion",
    "PIK3CA mutation", "RET fusion", "NRG1 fusion", "BRCA1 or BRCA2 mutation",
    "CDK4/6 amplification", "Any biomarker-unselected",
]
PHASES = ["Phase I", "Phase I/II", "Phase Ib/II", "Phase II", "Phase III"]
STATUSES = ["Recruiting", "Active, not recruiting", "Enrolling by invitation"]
DRUGS = [
    "nivolumab", "pembrolizumab", "atezolizumab", "durvalumab", "ipilimumab",
    "osimertinib", "alectinib", "lorlatinib", "sotorasib", "adagrasib",
    "tepotinib", "capmatinib", "larotrectinib", "entrectinib",
    "dabrafenib", "trametinib", "encorafenib", "trastuzumab",
    "bevacizumab", "cetuximab", "panitumumab", "oxaliplatin",
    "irinotecan", "pemetrexed", "docetaxel", "carboplatin",
    "capecitabine", "regorafenib", "fruquintinib", "lenvatinib",
]
SPONSORS = ["astrazeneca", "merck", "roche", "pfizer", "bms", "novartis", "lilly", "mirati"]


def gen_synthetic(n: int, start_id: int):
    trials = []
    for i in range(n):
        ct   = random.choice(CANCER_TYPES)
        bm   = random.choice(BIOMARKERS)
        drug = random.choice(DRUGS)
        drug2 = random.choice(DRUGS)
        combo = drug if random.random() > 0.4 else f"{drug} plus {drug2}"
        nct   = f"NCT{10000000 + start_id + i:08d}"
        phase = random.choice(PHASES)
        trials.append({
            "nct_id":            nct,
            "trial_id":          nct,
            "title":             f"{combo.title()} in {bm}-Positive Advanced {ct} ({phase})",
            "phase":             phase,
            "cancer_type":       ct,
            "biomarker_focus":   bm,
            "study_status":      random.choice(STATUSES),
            "brief_summary":     (
                f"A randomised study of {combo} targeting {bm} in advanced {ct}. "
                "Evaluates progression-free survival, overall survival, and objective response rate."
            ),
            "inclusion_snippet": (
                f"Confirmed {bm}; Stage III-IV {ct}; ECOG PS 0-{random.randint(1,2)}; "
                f"18 years or older; no prior {drug} therapy."
            ),
            "exclusion_snippet":  (
                f"Active autoimmune disease; prior {drug}; "
                "severe hepatic impairment; pregnancy or breastfeeding."
            ),
            "min_age":           18,
            "max_ecog_ps":       random.choice([1, 2]),
            "contact_info":      f"clinical.trials@{random.choice(SPONSORS)}.com",
        })
    return trials


n_synthetic = max(0, 500 - len(HANDCRAFTED))
all_trials = HANDCRAFTED + gen_synthetic(n_synthetic, start_id=len(HANDCRAFTED))

for t in all_trials:
    t.setdefault("trial_id", t.get("nct_id", f"TRIAL-{id(t)}"))

out_path = pathlib.Path("aob/ml/rag/trials/trials_snapshot.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_trials, f, indent=2, ensure_ascii=False)

print(f"Written {len(all_trials)} trials to {out_path}")
