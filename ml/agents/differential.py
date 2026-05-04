"""
ml/agents/differential.py
===========================
Differential Diagnosis Agent — produces top-3 candidate diagnoses with
probabilities from GigaPath softmax + VLM cross-check + metadata context.

This agent runs AFTER the Pathologist and BEFORE the Researcher.
Its output enriches the Oncologist's differential reasoning and signals
which alternative diagnoses require explicit rule-out tests.

Output schema:
    {
      "differentials": [
        {"diagnosis": str, "probability": float, "supporting_features": [str]},
        ...
      ],
      "rule_out_tests": [str],
      "primary_confidence": "high" | "moderate" | "low"
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from ml.models.llm_client import OllamaClient
from ml.agents.pathologist import PathologyReport

log = logging.getLogger(__name__)

DIFFERENTIAL_SYSTEM = """You are a senior diagnostic pathologist specialising in oncology.
You receive a structured pathology finding and must produce a differential diagnosis.
Output ONLY a JSON object with keys:
  differentials (list of up to 3 objects, each with: diagnosis (string), probability (float 0-1), supporting_features (list of strings)),
  rule_out_tests (list of IHC / molecular tests to confirm or exclude alternatives),
  primary_confidence ("high" if top diagnosis probability > 0.75, "moderate" if 0.5-0.75, "low" otherwise).
Order differentials by probability descending. Probabilities must sum to ≤1.0.
Output only valid JSON — no markdown fences, no preamble."""

# ── Known differential mappings by tissue type (for offline fallback) ─────────
_TISSUE_DIFFERENTIALS: dict[str, list[dict]] = {
    "lung_adenocarcinoma": [
        {"diagnosis": "Lung Adenocarcinoma", "supporting_features": ["glandular architecture", "TTF-1 positive", "Napsin A positive", "peripheral location"]},
        {"diagnosis": "Metastatic Adenocarcinoma", "supporting_features": ["multiple lesions", "prior GI/breast history", "CK7+/CK20+ pattern"]},
        {"diagnosis": "Adenocarcinoma in Situ (AIS)", "supporting_features": ["lepidic pattern predominant", "<3 cm", "no invasive component"]},
    ],
    "lung_squamous_cell_carcinoma": [
        {"diagnosis": "Lung Squamous Cell Carcinoma", "supporting_features": ["keratinisation", "p40/p63 positive", "CK5/6 positive", "central location"]},
        {"diagnosis": "Large Cell Carcinoma", "supporting_features": ["no squamous/glandular differentiation", "TTF-1 negative", "lacks clear lineage markers"]},
        {"diagnosis": "Poorly Differentiated Adenocarcinoma", "supporting_features": ["TTF-1 focally positive", "mucin staining", "peripheral location"]},
    ],
    "colon_adenocarcinoma": [
        {"diagnosis": "Colon Adenocarcinoma", "supporting_features": ["glandular structures", "CK20+", "CDX2+", "loss of nuclear polarity"]},
        {"diagnosis": "Metastatic Lung/Gastric Adenocarcinoma", "supporting_features": ["CK7+ / CK20- pattern", "TTF-1 or HER2+", "history of upper GI/lung primary"]},
        {"diagnosis": "High-Grade Dysplasia without Invasion", "supporting_features": ["no mucosal breakthrough", "confined to lamina propria", "no desmoplastic reaction"]},
    ],
    "lung_benign_tissue": [
        {"diagnosis": "Benign Lung Tissue / Reactive Change", "supporting_features": ["no nuclear atypia", "preserved alveolar architecture", "inflammatory infiltrate"]},
        {"diagnosis": "Organising Pneumonia", "supporting_features": ["Masson body formation", "intraalveolar polypoid plugs", "no malignant cells"]},
        {"diagnosis": "Atypical Adenomatous Hyperplasia (AAH)", "supporting_features": ["ground-glass opacity <5 mm", "mild nuclear atypia", "lepidic growth"]},
    ],
    "colon_benign_tissue": [
        {"diagnosis": "Benign Colonic Mucosa / Polyp", "supporting_features": ["preserved crypt architecture", "no nuclear atypia", "normal goblet cell density"]},
        {"diagnosis": "Tubular Adenoma", "supporting_features": ["crowded tubular glands", "pencillate nuclei", "mild dysplasia"]},
        {"diagnosis": "Hyperplastic Polyp", "supporting_features": ["serrated architecture", "elongated crypts", "no dysplasia"]},
    ],
}


@dataclass
class DifferentialResult:
    """Differential diagnosis result."""
    differentials:      list       # list[dict] with diagnosis/probability/supporting_features
    rule_out_tests:     list[str]
    primary_confidence: str        # "high" | "moderate" | "low"
    source:             str        # "llm" | "prototype"
    error:              Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def primary(self) -> Optional[dict]:
        return self.differentials[0] if self.differentials else None


def _fallback_differentials(tissue_type: str, pathology_confidence: float) -> DifferentialResult:
    """Return prototype-based differentials when LLM is unavailable."""
    template = _TISSUE_DIFFERENTIALS.get(tissue_type)
    if template is None:
        return DifferentialResult(
            differentials=[{"diagnosis": tissue_type.replace("_", " ").title(),
                            "probability": pathology_confidence,
                            "supporting_features": []}],
            rule_out_tests=[],
            primary_confidence="low",
            source="prototype",
            error="Unknown tissue type",
        )

    # Distribute probabilities: pathology_confidence for primary, rest split
    p0  = pathology_confidence
    rem = max(0.0, 1.0 - p0)
    probs = [p0]
    n_others = len(template) - 1
    if n_others > 0:
        probs += [round(rem / n_others, 3)] * n_others

    diffs = []
    for i, tmpl in enumerate(template):
        diffs.append({
            "diagnosis":          tmpl["diagnosis"],
            "probability":        probs[i] if i < len(probs) else 0.01,
            "supporting_features": tmpl["supporting_features"],
        })

    conf = "high" if p0 > 0.75 else ("moderate" if p0 > 0.5 else "low")
    rule_outs = {
        "lung_adenocarcinoma":          ["TTF-1 IHC", "Napsin A IHC", "CK7/CK20 panel"],
        "lung_squamous_cell_carcinoma": ["p40 IHC", "p63 IHC", "CK5/6 IHC", "TTF-1 exclusion"],
        "colon_adenocarcinoma":         ["CDX2 IHC", "CK20 IHC", "CK7 exclusion", "MLH1/MSH2 MMR IHC"],
        "lung_benign_tissue":           ["Ki-67 proliferation index", "TTF-1 (AIS exclusion)"],
        "colon_benign_tissue":          ["MMR IHC if family history", "Ki-67"],
    }.get(tissue_type, [])

    return DifferentialResult(
        differentials=diffs,
        rule_out_tests=rule_outs,
        primary_confidence=conf,
        source="prototype",
    )


class DifferentialDxAgent:
    """
    Differential Diagnosis Agent.

    Uses Llama 3.3 70B (via Ollama) to generate a differential diagnosis
    from the PathologyReport context. Falls back to prototype-based
    differentials if the LLM is unavailable.
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def analyse(
        self,
        pathology: PathologyReport,
        vlm_opinion=None,
        metadata: Optional[dict] = None,
    ) -> DifferentialResult:
        """
        Produce differential diagnosis for the given pathology report.

        Args:
            pathology:   PathologyReport from GigaPath.
            vlm_opinion: Optional VLMOpinion (Qwen2-VL second-opinion context).
            metadata:    Optional patient metadata dict.

        Returns:
            DifferentialResult with differentials + rule_out_tests.
        """
        # Build context for LLM
        context_parts = [
            f"Primary finding: {pathology.tissue_type.replace('_', ' ').title()} "
            f"(GigaPath confidence: {pathology.confidence:.1%})",
            f"Morphological features: {', '.join(getattr(pathology, 'morphological_features', []) or ['not specified'])}",
        ]

        if pathology.flags:
            context_parts.append(f"Flags: {', '.join(pathology.flags)}")

        if vlm_opinion is not None and hasattr(vlm_opinion, "suspected_tissue_type"):
            context_parts.append(f"VLM second opinion: {vlm_opinion.suspected_tissue_type}")
            if vlm_opinion.malignancy_indicators:
                context_parts.append(f"Malignancy indicators: {', '.join(vlm_opinion.malignancy_indicators[:3])}")

        if metadata:
            meta_items = []
            if "age" in metadata:
                meta_items.append(f"Age: {metadata['age']}")
            if "sex" in metadata:
                meta_items.append(f"Sex: {metadata['sex']}")
            if "smoking" in metadata:
                meta_items.append(f"Smoking: {metadata['smoking']}")
            if meta_items:
                context_parts.append("Patient: " + ", ".join(meta_items))

        context = ". ".join(context_parts)

        user_prompt = (
            f"Pathology finding:\n{context}\n\n"
            f"Produce a differential diagnosis with top-3 candidates, probabilities, "
            f"supporting histological features, and rule-out tests."
        )

        try:
            raw = self.llm.generate(
                system_prompt=DIFFERENTIAL_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=512,
            )
            result = self._parse_llm_response(raw)
            if result:
                return result
        except Exception as e:
            log.warning(f"DifferentialDxAgent LLM failed: {e} — using prototype fallback")

        return _fallback_differentials(pathology.tissue_type, pathology.confidence)

    def _parse_llm_response(self, raw: str) -> Optional[DifferentialResult]:
        """Parse LLM JSON output into DifferentialResult."""
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None or "differentials" not in parsed:
            return None

        diffs = parsed.get("differentials", [])
        if not isinstance(diffs, list):
            return None

        # Validate and normalise each differential
        clean_diffs = []
        for d in diffs[:3]:
            if not isinstance(d, dict):
                continue
            clean_diffs.append({
                "diagnosis":          str(d.get("diagnosis", "Unknown")),
                "probability":        float(d.get("probability", 0.0)),
                "supporting_features": list(d.get("supporting_features", [])),
            })

        if not clean_diffs:
            return None

        conf  = str(parsed.get("primary_confidence", "low"))
        tests = list(parsed.get("rule_out_tests", []))

        return DifferentialResult(
            differentials=clean_diffs,
            rule_out_tests=tests,
            primary_confidence=conf,
            source="llm",
        )
