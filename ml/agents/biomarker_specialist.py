"""
ml/agents/biomarker_specialist.py
===================================
Biomarker Specialist Agent — calls the fine-tuned biomarker LoRA adapter
served via vLLM's OpenAI-compatible endpoint.

Architecture
------------
This is Agent 2b in the AOB pipeline. It runs AFTER the TNM Staging Specialist
and BEFORE the Treatment Specialist and Oncologist.

Its output informs the Oncologist which molecular tests are required before
first-line therapy can be selected and which targeted/IO therapies are
"gated" pending biomarker results.

Output schema:
    {
      "tests_required": [...],
      "gated_therapies": [...],
      "rationale": "..."
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

# Prompt template must match exactly what was used for training biomarker_lora
PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified molecular oncologist.
Given a pathology description, output ONLY a JSON object with keys:
  tests_required (list of molecular/IHC tests),
  gated_therapies (list of targeted/immunotherapy options pending test results),
  rationale (one sentence citing NCCN guideline rationale).
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{pathology_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("BIOMARKER_VLLM_BASE_URL", "http://localhost:8006/v1")
_DEFAULT_MODEL    = os.getenv("BIOMARKER_VLLM_MODEL",    "biomarker_specialist")
_DEFAULT_TIMEOUT  = int(os.getenv("BIOMARKER_VLLM_TIMEOUT_S", "30"))

_REQUIRED_KEYS = {"tests_required", "gated_therapies", "rationale"}


@dataclass
class BiomarkerPanel:
    """Structured biomarker panel output from the specialist."""
    tests_required:   list[str]
    gated_therapies:  list[str]
    rationale:        str
    confidence:       str           # "high" | "low" | "unavailable"
    source:           str           # "biomarker_specialist" | "fallback"
    latency_ms:       float
    raw_output:       str
    error:            Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"

    def summary(self) -> str:
        n_tests = len(self.tests_required)
        n_gated = len(self.gated_therapies)
        return f"{n_tests} tests required, {n_gated} gated therapies"


def _fallback(error: str, latency_ms: float = 0.0) -> BiomarkerPanel:
    return BiomarkerPanel(
        tests_required=[], gated_therapies=[],
        rationale="Biomarker specialist unavailable — molecular workup deferred to Oncologist.",
        confidence="unavailable", source="fallback",
        latency_ms=latency_ms, raw_output="", error=error,
    )


class BiomarkerSpecialistAgent:
    """
    Biomarker Specialist — wraps the fine-tuned Llama-3-8B biomarker LoRA
    adapter served via vLLM.

    Converts a pathology text into a validated BiomarkerPanel specifying
    which molecular tests to order and which therapies are gated on results.
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
        log.info(f"BiomarkerSpecialistAgent: base_url={self.base_url} model={self.model}")

    def ping(self) -> bool:
        try:
            resp = self._client.get("/models", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["id"] for m in resp.json().get("data", [])]
            return self.model in models
        except Exception as e:
            log.debug(f"BiomarkerSpecialistAgent ping failed: {e}")
            return False

    def extract(self, pathology_text: str) -> BiomarkerPanel:
        """
        Extract biomarker panel recommendations from a pathology text.

        Args:
            pathology_text: Combined tissue type + stage description.

        Returns:
            BiomarkerPanel with tests_required, gated_therapies, rationale.
            Returns fallback on error.
        """
        if not pathology_text or not pathology_text.strip():
            return _fallback("Empty pathology text.")

        t0 = time.perf_counter()

        payload = {
            "model":    self.model,
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You are a board-certified molecular oncologist. "
                        "Given a pathology description, output ONLY a JSON object with keys: "
                        "tests_required (list), gated_therapies (list), rationale (string). "
                        "Output only valid JSON."
                    ),
                },
                {"role": "user", "content": pathology_text.strip()},
            ],
            "max_tokens":  256,
            "temperature": 0.0,
            "stop":        ["<|eot_id|>"],
        }

        raw_text  = ""
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
            return _fallback(last_error, latency_ms)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        result     = self._parse(raw_text, latency_ms)
        log.info(
            f"BiomarkerSpecialistAgent: {result.summary()} "
            f"confidence={result.confidence} latency={latency_ms}ms"
        )
        return result

    def _parse(self, raw_text: str, latency_ms: float) -> BiomarkerPanel:
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
            return BiomarkerPanel(
                tests_required=[], gated_therapies=[],
                rationale="Parse failed — biomarker info deferred to Oncologist.",
                confidence="low", source="biomarker_specialist",
                latency_ms=latency_ms, raw_output=raw_text,
                error="JSON parse failed",
            )

        missing = _REQUIRED_KEYS - set(parsed.keys())
        confidence = "low" if missing else "high"

        return BiomarkerPanel(
            tests_required=list(parsed.get("tests_required", [])),
            gated_therapies=list(parsed.get("gated_therapies", [])),
            rationale=str(parsed.get("rationale", "")),
            confidence=confidence,
            source="biomarker_specialist",
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
    p.add_argument("--text", default=(
        "Lung adenocarcinoma, Stage IV (T2a N2 M1b). "
        "EGFR/ALK/ROS1 pending. PD-L1 TPS 35%."
    ))
    args = p.parse_args()

    agent  = BiomarkerSpecialistAgent(base_url=args.base_url)
    panel  = agent.extract(args.text)
    print(json.dumps(panel.to_dict(), indent=2))
    agent._client.close()
