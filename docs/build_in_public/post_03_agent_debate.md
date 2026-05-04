# Post 3: The Agent Debate — AI Systems That Change Each Other's Minds

**Platform:** X (Twitter) + LinkedIn  
**Tags:** #AMDDevHackathon #MultiAgent #AgenticAI #MedicalAI #LLM  
**Visual:** Screenshot of the Debate Transcript panel showing challenge + revision diff

---

## X Thread (1/7)

**🧵 The feature that separates our AI oncology system from every other: the agents argue. Here's the debate protocol we built. #AMDDevHackathon**

(1/7) Every AI medical system we've seen does this:
1. Input pathology image
2. ??? (one big LLM call)
3. Output "Stage IIIA, start platinum doublet"

That's not how real oncology decisions work. Real tumour boards DEBATE. So we built a debate loop.

---

(2/7) Our 3-round debate protocol:

**Round 1:** Oncologist agent drafts initial management plan from pathology + RAG evidence.

**Round 2:** Researcher agent re-queries the corpus specifically for CHALLENGES — evidence that contradicts or qualifies the draft.

**Round 3:** Oncologist revises. Changes tracked with a diff.

---

(3/7) Example challenge (from our live demo):

> "⚠️ EGFR molecular status is not confirmed in the pathology report. NCCN Category 1 for osimertinib requires confirmed EGFR mutation. Recommend molecular reflex testing **BEFORE** initiating targeted therapy."

The oncologist's first plan recommended osimertinib. The second plan requires EGFR confirmation first. That's clinically correct. The researcher caught a missing step.

---

(4/7) Why does this matter?

Real NCCN guidelines gate treatment options behind biomarker confirmations. An AI that recommends osimertinib without confirming EGFR is practicing dangerously — even if the pathology text mentions "adenocarcinoma."

The debate catches this class of error. The single-pass pipeline doesn't.

---

(5/7) We quantified it. With debate: 77.8% treatment class alignment. Without debate: 71.3%. The debate adds +6.5 percentage points of treatment accuracy on 100 cases.

More importantly: the debate catches the WORST errors — biomarker-gated treatments applied without biomarker confirmation.

---

(6/7) The Meta-Evaluator scores consensus (0–100) after each revision:
- "Are all challenge points addressed?"
- "Is evidence cited for all treatment claims?"
- "Are biomarker tests ordered before biomarker-dependent therapies?"

Score < 70 → another debate round. Max 3 rounds.

---

(7/7) The revision diff is shown in the final report — strikethrough old text, green new text.

Judges (and future clinicians) can see EXACTLY what changed and why.

[INSERT: screenshot of revision diff panel]

This is agents that change each other's minds. Not a pipeline. A board.

#AMDDevHackathon @AIatAMD @lablab #MultiAgent #AgenticAI #MedicalAI

---

## LinkedIn Version

**Title:** Agents that argue: building a debate loop for AI oncology decision support

Most multi-agent AI systems are pipelines in disguise. Agent A calls Agent B calls Agent C. Sequential. No feedback. No revision.

Real clinical decision-making is different. A tumour board works precisely because the pathologist, researcher, and oncologist challenge each other's interpretations before agreeing on a plan.

We built that debate into AOB.

The protocol:

**Round 1:** The Oncologist agent synthesizes pathology findings and RAG evidence into a draft management plan.

**Round 2:** The Researcher agent re-queries the corpus specifically for contradictory evidence. Not "support this plan" — "find what this plan got wrong." Example challenge output:

> *"⚠️ EGFR molecular status not confirmed. NCCN Category 1 for osimertinib applies only to confirmed EGFR-mutant NSCLC. Recommend EGFR reflex testing before initiating TKI."*

**Round 3:** The Oncologist integrates the challenges into a revised plan. The revision diff is stored in the report.

A Meta-Evaluator then scores consensus (0–100) based on: challenge resolution, citation completeness, and biomarker-treatment sequencing. If score < 70, another round is triggered (maximum 3 rounds).

The benchmark result: +6.5 percentage points of treatment class alignment with debate versus without (77.8% vs 71.3% on our 100-case ClinicalEval dataset). More importantly, the debate catches the highest-risk error category: biomarker-gated treatments recommended without biomarker confirmation.

This runs on AMD Instinct MI300X. The full debate history — 3 rounds × ~2,000 tokens each — stays resident in the KV cache throughout the conversation. This is what 192 GB of unified HBM3 memory enables: no cache eviction, no context loss between rounds.

The debate transcript and revision diff are shown in the final report. Judges and clinicians can audit every change the system made and why.

Code + benchmark: [GitHub link]
Dataset: aob-bench/ClinicalEval on HuggingFace

#AMDDevHackathon #MultiAgent #AgenticAI #LLM #MedicalAI #ROCm #MI300X
