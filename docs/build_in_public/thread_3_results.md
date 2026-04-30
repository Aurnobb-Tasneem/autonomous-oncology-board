# Thread 3: Results — 3-Agent Board Debate In Action
**Build in Public — Autonomous Oncology Board | @AurnabbTasneem**

---

**1/12**
We ran our 3-agent AI oncology board on real demo cases.

Here's what actually happened when the agents debated — including the cases where they disagreed, revised, and reached consensus 🧵

---

**2/12**
Quick context on what we built:

→ **Agent 1 (Pathologist):** GigaPath ViT-Giant analyses histopathology slides
→ **Agent 2 (Researcher):** RAG over NCCN guidelines
→ **Agent 3 (Oncologist):** Llama 3.3 70B synthesises final plan
→ Running on AMD MI300X (192 GB HBM3)

The agents debate each case for up to 3 rounds.

---

**3/12**
**Case 1: Lung Adenocarcinoma (67M, non-smoker)**

**Pathologist report:**
- 6 patches → Lung Adenocarcinoma (87% confidence)
- Mean abnormality score: 0.63
- Biomarkers: High nuclear pleomorphism, High mitotic index
- MC Dropout uncertainty: ±3.8% (low uncertainty)

Flags: high_abnormality_detected

---

**4/12**
**Oncologist initial plan:**
- Stage: IV NSCLC (adenocarcinoma) — pending molecular workup
- First-line: **Carboplatin + Pemetrexed**
- Immediate actions: CT staging, molecular panel, MDT review

Seems reasonable. But then the Researcher challenged it.

---

**5/12**
**Researcher Challenge (Round 1):**

> "NCCN 2024: For non-squamous NSCLC in non-smokers, EGFR testing is mandatory before initiating platinum-based chemotherapy. 
> EGFR exon 19/21 mutations are present in 40-50% of non-smoker adenocarcinomas.
> First-line osimertinib (if EGFR+) has superior PFS vs. chemotherapy."

Flagged issues:
- EGFR testing not listed before starting chemotherapy
- Missing ALK, ROS1, MET, KRAS panels
- Non-smoker population → higher actionable mutation probability

---

**6/12**
**Oncologist Revision:**

> "✅ REVISED: Carboplatin/pemetrexed now gated on EGFR/ALK/ROS1/MET/KRAS results.
> First-line changed to: **Comprehensive molecular panel FIRST, then osimertinib if EGFR+ / pembrolizumab monotherapy if PD-L1 >50% / carboplatin+pemetrexed+pembrolizumab if driver-negative**"

**MetaEvaluator consensus score: 88/100** ✅ Debate concluded in Round 1.

---

**7/12**
**Case 2: Colon Adenocarcinoma (58F, T3N1M0)**

**Pathologist report:**
- 6 patches → Colon Adenocarcinoma (91% confidence)
- Mean abnormality score: 0.71
- Biomarkers: High necrosis extent, Low gland formation
- MC Dropout: ±4.1% (low uncertainty)

Flags: high_abnormality_detected, heterogeneous_tissue

---

**8/12**
**Initial plan → Researcher challenge:**

The initial plan proposed FOLFOX as first-line without mentioning MSI status.

Researcher challenge:
> "MSI-H colorectal cancer (15% of Stage III cases) has dramatically different treatment — adjuvant pembrolizumab vs. traditional FOLFOX.
> KRAS/NRAS mutation status also gates cetuximab eligibility.
> MSS vs. MSI-H distinction is mandatory before finalising adjuvant therapy."

Consensus score after revision: **79/100** ✅

---

**9/12**
**Case 3: Lung Squamous Cell (72M, 50 pack-year smoker)**

This one triggered 2 debate rounds.

After Round 1 revision (PD-L1 gating added), consensus score: **61/100** — below threshold.

The Researcher wasn't satisfied: "Stage IIIB disease requires radiation oncology involvement before systemic therapy decision."

---

**10/12**
Round 2 added:
→ Definitive chemoradiation (concurrent carboplatin+paclitaxel + RT) as first-line for unresectable Stage III
→ Durvalumab consolidation post-CRT (PACIFIC regimen)
→ Radiation oncology referral made explicit

**Round 2 consensus score: 84/100** ✅ Done.

Total time for this case: 4m 12s (including both debate rounds).

---

**11/12**
What the debate catches that a single-pass model misses:

| Issue Caught | Without Debate | With Debate |
|---|---|---|
| Missing EGFR testing before chemo | ❌ | ✅ Fixed Round 1 |
| MSI-H immunotherapy eligibility | ❌ | ✅ Fixed Round 1 |
| Missing RT for Stage III | ❌ | ✅ Fixed Round 2 |
| KRAS/NRAS gating for anti-EGFR | ❌ | ✅ Fixed Round 1 |

3 cases. 4 critical clinical gaps caught. All before a clinician reviewed.

---

**12/12**
The system isn't making clinical decisions — it's catching what a single AI pass misses.

Think of it as a second (and third) opinion, automated, grounded in NCCN 2024 guidelines, running in under 5 minutes.

Full code, open-source:
👉 github.com/Aurnobb-Tasneem/autonomous-oncology-board

Try the live demo at the HuggingFace Space link in the README.

#MedicalAI #Oncology #MultiAgent #AMD #BuildInPublic #LLM
