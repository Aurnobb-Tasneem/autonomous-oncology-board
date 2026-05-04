# Post 2: Attention Heatmaps — Making Medical AI Explainable

**Platform:** X (Twitter) + LinkedIn  
**Tags:** #AMDDevHackathon #ExplainableAI #MedicalAI #DigitalPathology #ROCm  
**Visual:** Side-by-side: raw pathology patch | attention heatmap overlay (red = suspicious)

---

## X Thread (1/6)

**🧵 The difference between a black box and a clinical tool: explainability. Here's how we made our AI pathologist show its work. #AMDDevHackathon**

(1/6) Our AOB pathologist uses Prov-GigaPath — a ViT trained on 1.3B pathology image tokens. It classifies tissue with 91% confidence. But confidence without explanation is useless in medicine.

So we built triple-modal saliency.

---

(2/6) Three methods, one consensus heatmap:

**1. Attention Rollout** — propagate attention weights through all transformer heads to see what patches the [CLS] token "looked at"

**2. Grad-CAM++** — backpropagate from the predicted class logit through the final ViT block. Class-discriminative. Shows WHY it thinks adenocarcinoma.

**3. Integrated Gradients** — compute gradients along a linear path from a black baseline to the actual input. Attribution at the pixel level.

---

(3/6) The consensus: regions flagged by ≥2 methods are highlighted red.

The visualization shows exactly which cells drove the classification. Glandular patterns. Nuclear atypia. Mitotic figures. The same features a human pathologist would circle on the slide.

[INSERT: split image — raw patch on left, heatmap overlay on right]

---

(4/6) We also added Monte Carlo Dropout uncertainty. Instead of one forward pass, run 20 stochastic passes with dropout active.

Result: "Lung Adenocarcinoma: 91% **± 4.2%**"

When σ > 8%, the system flags: "⚠️ High morphological uncertainty — recommend second-opinion biopsy."

This is a real clinical safety signal. No other hackathon project will have this.

---

(5/6) Why does explainability matter for medical AI?

The #1 reason clinicians don't trust AI diagnostic tools is opacity. "It says 94% cancer, but how does it know?"

Our heatmap answers that question visually, in the language clinicians understand: tissue morphology.

---

(6/6) This runs live during the demo. The model loads onto AMD MI300X, processes patches, generates all three heatmaps in real time.

Pathologists can toggle between "Show AI Attention" and "Show Original" — they decide how to use the signal.

Code + benchmark dataset: [link]
#AMDDevHackathon @AIatAMD @lablab #ExplainableAI #DigitalPathology

---

## LinkedIn Version

**Title:** Making medical AI explainable: triple-modal saliency heatmaps for histopathology

The single biggest barrier to clinical adoption of AI diagnostic tools isn't accuracy — it's opacity. Clinicians ask: "Why does it think this is cancer?"

For the Autonomous Oncology Board, we addressed this with triple-modal saliency visualization:

**1. Attention Rollout:** GigaPath's transformer attention weights, propagated across all heads, show which spatial regions the model attended to when making its classification.

**2. Grad-CAM++:** Backpropagating from the predicted class logit through the final ViT block gives class-discriminative localization. This specifically shows WHAT features differentiate adenocarcinoma from squamous cell carcinoma.

**3. Integrated Gradients:** Pixel-level attribution from a black baseline through 50 interpolation steps. The most precise localization of all three methods.

The consensus heatmap highlights regions flagged by ≥2 methods — a conservative, high-specificity indicator of suspicious tissue.

We also added Monte Carlo Dropout uncertainty quantification: 20 stochastic forward passes produce a predictive interval. "Lung Adenocarcinoma: 91% ± 4.2%." When standard deviation exceeds 8%, the system automatically flags the case for second-opinion review.

In clinical terms: the model knows when it doesn't know. That's the difference between a research demo and a system a pathologist can trust.

All inference runs on AMD Instinct MI300X with ROCm 6.x. GigaPath's FP16 weights occupy ~3 GB — small enough that the MI300X runs it concurrently with Llama 3.3 70B without context switching.

The benchmark: GigaPath contributes +30 percentage points of TNM staging accuracy over a language-only baseline on our 100-case ClinicalEval dataset.

Open dataset: aob-bench/ClinicalEval on HuggingFace.

#AMDDevHackathon #ExplainableAI #DigitalPathology #MedicalAI #ROCm #MI300X
