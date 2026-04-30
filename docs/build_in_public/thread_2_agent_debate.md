# Thread 2: How We Built Multi-Agent Debate for Oncology AI
**Build in Public — Autonomous Oncology Board | @AurnabbTasneem**

---

**1/14**
Single-model AI makes confident mistakes.

A senior oncologist doesn't just diagnose — they argue, challenge, revise, and reach consensus with colleagues.

We built that exact process into our AI system. Here's how multi-agent debate works in the AOB 🧵

---

**2/14**
The core insight: in medicine, disagreement is a feature, not a bug.

A good tumour board meeting sounds like:
→ "The morphology suggests adenocarcinoma"
→ "But the Ki-67 is unusually high — could be small cell"
→ "The gland formation pattern rules that out"
→ "Agreed, but let's gate therapy on EGFR result"

We modelled exactly this.

---

**3/14**
Our three agents:

**Agent 1 — Pathologist (GigaPath)**
Analyses histopathology slides. Returns: tissue type, confidence, abnormality scores, heatmaps.

**Agent 2 — Researcher (RAG + Llama 3.3 70B)**
Retrieves NCCN guideline evidence. Returns: treatment options, evidence quality, citations.

**Agent 3 — Oncologist (Llama 3.3 70B)**
Synthesises agents 1+2 into a complete management plan.

---

**4/14**
The standard pipeline (no debate):

```
Pathologist → Researcher → Oncologist → Plan
```

Takes ~45-90s. Good for clear-cut cases.

But what about ambiguous cases? Cases where the first plan might miss something?

---

**5/14**
That's where the Debate Loop kicks in.

After the initial plan, the Researcher gets a second job:
**Challenge the plan against NCCN guidelines.**

Specifically:
→ Is the proposed regimen evidence-based?
→ Are the required molecular tests ordered?
→ Does the stage align with the morphology?

---

**6/14**
The challenge prompt:

```python
"You are a critical reviewer. Read the draft management plan.
 Identify any deviations from NCCN guidelines.
 Flag: missing molecular tests, contraindicated regimens,
 staging inconsistencies, or missing referrals.
 Return: challenge_text, flagged_issues[], severity."
```

The Researcher acts as devil's advocate to its own research.

---

**7/14**
If the Researcher flags morphological doubts, the Pathologist becomes a referee.

It re-examines its own findings:
→ Inter-patch consistency (% of patches agreeing on the dominant class)
→ Mean confidence vs. abnormality score correlation
→ Heterogeneity flags

Returns: morphology_confirmed (bool) + referee_note.

---

**8/14**
The Oncologist then revises:

```python
"ORIGINAL PLAN: [first-line treatment]
 RESEARCHER CHALLENGE: [critique]
 PATHOLOGIST REFEREE: [morphology re-evaluation]

 Revise your plan to address these concerns.
 Gate therapy on molecular results."
```

The revision is explicit — what changed and why is tracked in revision_notes.

---

**9/14**
Then the MetaEvaluator scores consensus (0–100):

→ Did the revision address the flagged issues?
→ Is the revised first-line evidence-based?
→ Are the immediate actions consistent with the stage?

Score < 70 → another debate round (max 3 rounds).
Score ≥ 70 → consensus reached, finalise plan.

---

**10/14**
Why a score threshold and not a binary pass/fail?

Because medical decisions exist on a spectrum.

A score of 65 means "mostly aligned, minor gaps" — acceptable to proceed.
A score of 40 means "fundamental disagreement" — needs another round.

The threshold (70) is configurable in `board.py`.

---

**11/14**
What does the debate transcript look like?

```json
{
  "round": 1,
  "researcher_challenge": "EGFR testing not ordered before initiating chemotherapy",
  "pathologist_referee": "Morphology confirms adenocarcinoma in 83% of patches",
  "oncologist_revision": "Added EGFR/ALK/ROS1 panel before initiating carboplatin",
  "consensus_score": 82,
  "revision_notes": "✅ REVISED: Gated chemotherapy on EGFR molecular result"
}
```

Full audit trail. Every change justified.

---

**12/14**
In practice, most cases reach consensus in Round 1.

Round 1 consensus rate: ~70% of cases
Round 2 needed: ~25% of cases
Round 3 (max): ~5% of cases — high complexity or truly ambiguous histology

The debate usually takes 2-3x longer than the standard pipeline.
Worth it for edge cases.

---

**13/14**
The debate architecture in code:

```
board.py → _run_debate()
  for round in 1..MAX_DEBATE_ROUNDS:
    researcher.challenge(draft_plan)    → critique
    pathologist.referee(flagged_issues)  → morphology_update (if needed)
    oncologist.revise(plan, critique)    → revised_plan
    meta_evaluator.evaluate(...)         → consensus_score
    if score >= 70: break
```

Under 100 lines of clean Python. No framework magic.

---

**14/14**
The key design decision: we didn't use CrewAI or AutoGen.

We wrote a hand-rolled Python state machine.

Why?
→ Full debuggability on ROCm
→ No hidden API calls or retry logic
→ Explicit step ordering for SSE streaming
→ Easier to demo and explain

Sometimes the boring solution is the right solution.

Code: github.com/Aurnobb-Tasneem/autonomous-oncology-board

#MultiAgent #MedicalAI #LLM #AMD #BuildInPublic
