"""
ml/agents/treatment_specialist.py
===================================
Treatment Specialist Agent — calls the fine-tuned treatment LoRA adapter
served via vLLM's OpenAI-compatible endpoint.

Architecture
------------
Agent 2c in the AOB pipeline. Runs AFTER Biomarker Specialist and
BEFORE the Oncologist. Provides an evidence-based initial treatment
proposal that the Oncologist then synthesises into the full Management Plan.

Output schema:
    {
      "first_line": "...",
      "second_line": "...",
      "nccn_category": "1" | "2A" | "2B",
      "contraindications": [...],
      "monitoring": [...]
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional

import httpx

PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified oncologist and NCCN guideline expert.
Given a clinical case description, output ONLY a JSON object with keys:
  first_line (recommended first-line treatment regimen),
  second_line (recommended second-line regimen),
  nccn_category ("1", "2A", or "2B"),
  contraindications (list),
  monitoring (list).
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{clinical_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("TREATMENT_VLLM_BASE_URL", "http://localhost:8006/v1")
_DEFAULT_MODEL    = os.getenv("TREATMENT_VLLM_MODEL",    "treatment_specialist")
_DEFAULT_TIMEOUT  = int(os.getenv("TREATMENT_VLLM_TIMEOUT_S", "30"))

_REQUIRED_KEYS = {"first_line", "second_line", "nccn_category", "contraindications", "monitoring"}
_VALID_NCCN_CATS = {"1", "2A", "2B", "2a", "2b"}


@dataclass
class TreatmentProposal:
    """Structured treatment plan from the Treatment Specialist."""
    first_line:       str
    second_line:      str
    nccn_category:    str        # "1" | "2A" | "2B"
    contraindications: list[str]
    monitoring:       list[str]
    confidence:       str        # "high" | "low" | "unavailable"
    source:           str        # "treatment_specialist" | "fallback"
    latency_ms:       float
    raw_output:       str
    error:            Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"

    def summary(self) -> str:
        return f"NCCN Cat {self.nccn_category}: {self.first_line[:60]}..."


def _fallback(error: str, latency_ms: float = 0.0) -> TreatmentProposal:
    return TreatmentProposal(
        first_line="Specialist unavailable — see Oncologist synthesis",
        second_line="", nccn_category="?",
        contraindications=[], monitoring=[],
        confidence="unavailable", source="fallback",
        latency_ms=latency_ms, raw_output="", error=error,
    )


class TreatmentSpecialistAgent:
    """
    Treatment Specialist — wraps the fine-tuned Llama-3-8B treatment LoRA
    adapter served via vLLM.

    Converts a clinical case description (tissue + stage + biomarkers) into
    a validated TreatmentProposal with first/second-line regimens, NCCN
    evidence category, contraindications, and monitoring parameters.
    """

    def __init__(
        self,
        base_url:    str  = _DEFAULT_BASE_URL,
        model:       str  = _DEFAULT_MODEL,
        api_key:     Optional[str] = None,
        timeout:     int  = _DEFAULT_TIMEOUT,
        max_retries: int  = 2,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout
        )
        log.info(f"TreatmentSpecialistAgent: base_url={self.base_url} model={self.model}")

    def ping(self) -> bool:
        try:
            resp = self._client.get("/models", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["id"] for m in resp.json().get("data", [])]
            return self.model in models
        except Exception:
            return False

    def plan(
        self,
        tissue_type: str,
        tnm_stage: str,
        biomarker_summary: str = "",
        metadata: Optional[dict] = None,
    ) -> TreatmentProposal:
        """
        Generate a treatment proposal for the given clinical profile.

        Args:
            tissue_type:       e.g. "Lung Adenocarcinoma"
            tnm_stage:         e.g. "Stage IIB (T2a N1 M0)"
            biomarker_summary: e.g. "EGFR exon 19 deletion. PD-L1 TPS 30%."
            metadata:          Optional patient metadata dict.

        Returns:
            TreatmentProposal — falls back gracefully if service unavailable.
        """
        meta_str = ""
        if metadata:
            parts = []
            if "age" in metadata:
                parts.append(f"Age {metadata['age']}")
            if "ecog_ps" in metadata:
                parts.append(f"ECOG PS {metadata['ecog_ps']}")
            if "smoking" in metadata:
                parts.append(f"Smoking: {metadata['smoking']}")
            if parts:
                meta_str = ". " + ". ".join(parts) + "."

        clinical_text = (
            f"{tissue_type}. {tnm_stage}. "
            f"{biomarker_summary}{meta_str}"
        ).strip()

        t0 = time.perf_counter()

        payload = {
            "model":    self.model,
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You are a board-certified oncologist and NCCN guideline expert. "
                        "Given a clinical case description, output ONLY a JSON object with keys: "
                        "first_line, second_line, nccn_category (1/2A/2B), contraindications (list), monitoring (list). "
                        "Output only valid JSON."
                    ),
                },
                {"role": "user", "content": clinical_text},
            ],
            "max_tokens":  512,
            "temperature": 0.0,
            "stop":        ["<|eot_id|>"],
        }

        raw_text   = ""
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                raw_text = (
                    resp.json()["choices"][0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                break
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        else:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            log.warning(f"TreatmentSpecialistAgent failed: {last_error}")
            return _fallback(last_error, latency_ms)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        result     = self._parse(raw_text, latency_ms)
        log.info(
            f"TreatmentSpecialistAgent: NCCN Cat {result.nccn_category} "
            f"confidence={result.confidence} latency={latency_ms}ms"
        )
        return result

    def _parse(self, raw_text: str, latency_ms: float) -> TreatmentProposal:
        parsed = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        if parsed is None:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            return TreatmentProposal(
                first_line="JSON parse failed — see Oncologist synthesis",
                second_line="", nccn_category="?",
                contraindications=[], monitoring=[],
                confidence="low", source="treatment_specialist",
                latency_ms=latency_ms, raw_output=raw_text,
                error="JSON parse failed",
            )

        missing    = _REQUIRED_KEYS - set(parsed.keys())
        nccn_cat   = str(parsed.get("nccn_category", "?")).upper()
        confidence = "high" if not missing and nccn_cat in {"1", "2A", "2B"} else "low"

        return TreatmentProposal(
            first_line=str(parsed.get("first_line", "")),
            second_line=str(parsed.get("second_line", "")),
            nccn_category=nccn_cat,
            contraindications=list(parsed.get("contraindications", [])),
            monitoring=list(parsed.get("monitoring", [])),
            confidence=confidence,
            source="treatment_specialist",
            latency_ms=latency_ms,
            raw_output=raw_text,
            error=f"Missing keys: {missing}" if missing else None,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._client.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default=_DEFAULT_BASE_URL)
    args = p.parse_args()

    agent    = TreatmentSpecialistAgent(base_url=args.base_url)
    proposal = agent.plan(
        tissue_type="Lung Adenocarcinoma",
        tnm_stage="Stage IV (T2a N2 M1b)",
        biomarker_summary="EGFR exon 19 deletion confirmed. PD-L1 TPS 30%. ALK/ROS1 negative.",
    )
    print(json.dumps(proposal.to_dict(), indent=2))
    agent._client.close()
