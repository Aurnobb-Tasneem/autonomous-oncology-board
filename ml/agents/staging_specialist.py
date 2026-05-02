"""
ml/agents/staging_specialist.py
================================
TNM Staging Specialist Agent — calls the fine-tuned LoRA adapter served via
vLLM's OpenAI-compatible endpoint (POST /v1/chat/completions).

Architecture
------------
This is Agent 0.5 in the AOB pipeline — an optional pre-processing step that
runs BEFORE the Researcher and Oncologist.  Its output supplements the
Pathologist report with a structured, model-grounded TNM stage, giving the
Oncologist a second opinion on staging rather than relying solely on RAG.

The LoRA adapter is trained by scripts/finetune_tnm.py (Track 2 deliverable).
It is served on a separate port (default 8006) via:

    bash scripts/serve_tnm_adapter.sh

or manually:

    python -m vllm.entrypoints.openai.api_server \\
      --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --enable-lora \\
      --lora-modules tnm_specialist=aob/ml/models/checkpoints/tnm_lora \\
      --port 8006 \\
      --gpu-memory-utilization 0.15 \\
      --dtype bfloat16

Integration into board.py (Task 3)
-----------------------------------
    from ml.agents.staging_specialist import StagingSpecialistAgent, TNMResult

    staging = StagingSpecialistAgent()      # uses env vars for config
    tnm = staging.stage(pathology.summary)  # fast — usually <2s on MI300X
    # tnm.to_dict() → {"T": "T2a", "N": "N1", "M": "M0", "stage": "IIB",
    #                   "confidence": "high", "source": "tnm_specialist"}

Design goals
------------
1. Fail gracefully: if vLLM is down, return a degraded result rather than
   crashing the board pipeline.
2. Validate strictly: parse and validate JSON — never pass unstructured text
   downstream.
3. Retry with exponential back-off: network blips on the MI300X host are rare
   but possible when vLLM is still warming up.
4. Log every call for the VRAM + latency dashboard.
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

# Prompt template — MUST match PROMPT_TEMPLATE in scripts/finetune_tnm.py exactly.
# The LoRA adapter was trained on this format; any deviation will degrade output.
# If you change the template here, retrain the adapter.
PROMPT_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified oncological pathologist.
Given a pathology text, output ONLY a JSON object with keys:
  T (tumour extent), N (node involvement), M (metastasis), stage (overall AJCC stage).
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{pathology_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

log = logging.getLogger(__name__)

# ── Defaults (override with environment variables) ───────────────────────────
_DEFAULT_BASE_URL = os.getenv("TNM_VLLM_BASE_URL", "http://localhost:8006/v1")
_DEFAULT_MODEL    = os.getenv("TNM_VLLM_MODEL",    "tnm_specialist")
_DEFAULT_TIMEOUT  = int(os.getenv("TNM_VLLM_TIMEOUT_S", "30"))

# TNM keys we require in every valid response
_REQUIRED_KEYS = {"T", "N", "M", "stage"}


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class TNMResult:
    """Structured TNM staging output from the fine-tuned specialist."""
    T:          str           # Tumour extent, e.g. "T2a"
    N:          str           # Node involvement, e.g. "N1"
    M:          str           # Metastasis, e.g. "M0"
    stage:      str           # Overall AJCC stage, e.g. "IIB"
    confidence: str           # "high" | "low" | "unavailable"
    source:     str           # "tnm_specialist" | "fallback"
    latency_ms: float         # Round-trip latency to the vLLM endpoint
    raw_output: str           # Raw model text (for debugging/audit)
    error:      Optional[str] # Set when source == "fallback"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"

    def tnm_string(self) -> str:
        """Human-readable TNM string, e.g. 'T2a N1 M0 (Stage IIB)'."""
        return f"{self.T} {self.N} {self.M} (Stage {self.stage})"


# ── Fallback result ───────────────────────────────────────────────────────────

def _fallback(error: str, latency_ms: float = 0.0) -> TNMResult:
    """
    Return a degraded result when the specialist endpoint is unavailable.

    Downstream agents (Oncologist, Researcher) must check `is_fallback`
    and treat this as "staging pending — not model-grounded".
    """
    return TNMResult(
        T="TX", N="NX", M="MX",
        stage="Undetermined",
        confidence="unavailable",
        source="fallback",
        latency_ms=latency_ms,
        raw_output="",
        error=error,
    )


# ── Agent ────────────────────────────────────────────────────────────────────

class StagingSpecialistAgent:
    """
    TNM Staging Specialist — wraps the fine-tuned Llama-3-8B LoRA adapter
    served via vLLM's OpenAI-compatible API.

    The agent converts a pathology text string into a validated TNMResult.
    It is intentionally stateless; the same instance can process multiple
    cases concurrently.

    Args:
        base_url:  Base URL of the vLLM OpenAI-compatible server.
                   Default: TNM_VLLM_BASE_URL env var or http://localhost:8006/v1
        model:     LoRA module alias registered with vLLM --lora-modules.
                   Default: TNM_VLLM_MODEL env var or "tnm_specialist"
        api_key:   Optional Bearer token (vLLM doesn't require one by default).
        timeout:   HTTP request timeout in seconds.
        max_retries: Number of retries on transient network errors.
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

        # Persistent httpx client (connection pooling)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )
        log.info(
            f"StagingSpecialistAgent: base_url={self.base_url}  "
            f"model={self.model}  timeout={self.timeout}s"
        )

    # ── Health check ─────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the vLLM server is reachable and the model is loaded."""
        try:
            resp = self._client.get("/models", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["id"] for m in resp.json().get("data", [])]
            available = self.model in models
            if not available:
                log.warning(
                    f"StagingSpecialistAgent: model '{self.model}' not in vLLM model list: "
                    f"{models}. Is the adapter registered with --lora-modules?"
                )
            return available
        except Exception as e:
            log.debug(f"StagingSpecialistAgent: ping failed — {e}")
            return False

    # ── Core staging call ─────────────────────────────────────────────────────

    def stage(self, pathology_text: str) -> TNMResult:
        """
        Stage a patient from a pathology text description.

        Sends the text to the fine-tuned TNM specialist and parses the
        structured JSON response into a validated TNMResult.

        Args:
            pathology_text: Free-text pathology finding, e.g.:
                "3.2 cm lung adenocarcinoma, 2/15 lymph nodes positive, no metastasis."

        Returns:
            TNMResult with T, N, M, stage fields.
            On error, returns a degraded TNMResult with source="fallback".
        """
        if not pathology_text or not pathology_text.strip():
            return _fallback("Empty pathology text provided.")

        t0 = time.perf_counter()

        # Build the exact prompt format the adapter was trained on
        user_message = pathology_text.strip()

        payload = {
            "model":       self.model,
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You are a board-certified oncological pathologist. "
                        "Given a pathology text, output ONLY a JSON object with keys: "
                        "T (tumour extent), N (node involvement), M (metastasis), "
                        "stage (overall AJCC stage). "
                        "Do not output any explanation. Output only valid JSON."
                    ),
                },
                {
                    "role":    "user",
                    "content": user_message,
                },
            ],
            "max_tokens":  80,
            "temperature": 0.0,   # greedy — deterministic staging
            "stop":        ["<|eot_id|>"],
        }

        raw_text = ""
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_text = (
                    data["choices"][0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                break  # success

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                log.warning(
                    f"StagingSpecialistAgent: HTTP error on attempt {attempt+1}: {last_error}"
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # 1s, 2s back-off

            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                log.warning(
                    f"StagingSpecialistAgent: request error on attempt {attempt+1}: {last_error}"
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        else:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            log.error(
                f"StagingSpecialistAgent: all {self.max_retries + 1} attempts failed. "
                f"Last error: {last_error}"
            )
            return _fallback(last_error, latency_ms)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── Parse and validate JSON ──────────────────────────────────────────
        result = self._parse_tnm_json(raw_text, latency_ms)

        log.info(
            f"StagingSpecialistAgent: {result.tnm_string()}  "
            f"confidence={result.confidence}  latency={latency_ms}ms"
        )
        return result

    # ── JSON parsing ─────────────────────────────────────────────────────────

    def _parse_tnm_json(self, raw_text: str, latency_ms: float) -> TNMResult:
        """
        Extract and validate the TNM JSON from the model's raw output.

        Strategy:
          1. Try to parse the whole response as JSON.
          2. If that fails, extract the first {...} block and parse that.
          3. Validate that all four required keys are present.
          4. On any failure, return a fallback result.
        """
        parsed = None

        # Strategy 1: direct parse
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first JSON object
        if parsed is None:
            m = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            log.warning(
                f"StagingSpecialistAgent: could not extract JSON from output: "
                f"{raw_text[:200]!r}"
            )
            return TNMResult(
                T="TX", N="NX", M="MX",
                stage="Undetermined",
                confidence="low",
                source="tnm_specialist",
                latency_ms=latency_ms,
                raw_output=raw_text,
                error="JSON parse failed",
            )

        # Validate required keys
        missing = _REQUIRED_KEYS - set(parsed.keys())
        if missing:
            log.warning(
                f"StagingSpecialistAgent: JSON missing required keys {missing}: {parsed}"
            )
            return TNMResult(
                T=parsed.get("T", "TX"),
                N=parsed.get("N", "NX"),
                M=parsed.get("M", "MX"),
                stage=parsed.get("stage", "Undetermined"),
                confidence="low",
                source="tnm_specialist",
                latency_ms=latency_ms,
                raw_output=raw_text,
                error=f"Missing keys: {missing}",
            )

        # Clean and normalise values
        t_val     = str(parsed["T"]).strip()
        n_val     = str(parsed["N"]).strip()
        m_val     = str(parsed["M"]).strip()
        stage_val = str(parsed["stage"]).strip()

        # Confidence heuristic: high if all values look like real TNM codes
        confidence = _assess_confidence(t_val, n_val, m_val, stage_val)

        return TNMResult(
            T=t_val,
            N=n_val,
            M=m_val,
            stage=stage_val,
            confidence=confidence,
            source="tnm_specialist",
            latency_ms=latency_ms,
            raw_output=raw_text,
            error=None,
        )

    # ── Batch staging ─────────────────────────────────────────────────────────

    def stage_batch(self, pathology_texts: list[str]) -> list[TNMResult]:
        """
        Stage multiple cases sequentially.

        For the hackathon demo, sequential is sufficient (latency ~2s/case).
        In production, replace with concurrent httpx calls.

        Args:
            pathology_texts: List of pathology descriptions.

        Returns:
            List of TNMResult (same order as input).
        """
        return [self.stage(text) for text in pathology_texts]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._client.close()


# ── Confidence heuristic ─────────────────────────────────────────────────────

# Valid TNM prefixes (non-exhaustive but covers all LC25000 cases)
_T_PATTERN = re.compile(
    r"^(T[0-4][a-d]?|Tis(\(AIS\))?|T1mi|T[Xx0])$", re.IGNORECASE
)
_N_PATTERN = re.compile(r"^(N[0-3][a-c]?|N[Xx])$", re.IGNORECASE)
_M_PATTERN = re.compile(r"^(M[01][a-c]?|M[Xx])$", re.IGNORECASE)
_STAGE_PATTERN = re.compile(
    r"^(0|I[ABC]?[1-3]?|II[ABC]?|III[ABC]?|IV[ABC]?|N/A|Undetermined)$",
    re.IGNORECASE,
)


def _assess_confidence(T: str, N: str, M: str, stage: str) -> str:
    """
    Return 'high' if all four values match expected TNM patterns, else 'low'.

    This is a heuristic — it catches hallucinated free-text but is not a
    clinical validator.
    """
    if (
        _T_PATTERN.match(T)
        and _N_PATTERN.match(N)
        and _M_PATTERN.match(M)
        and _STAGE_PATTERN.match(stage)
    ):
        return "high"
    return "low"


# ── Module-level convenience function ────────────────────────────────────────

def stage_from_pathology_report(
    pathology_summary: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> TNMResult:
    """
    One-shot staging from a pathology summary string.

    Creates a temporary StagingSpecialistAgent, calls stage(), and closes
    the HTTP client.  Use StagingSpecialistAgent directly when making multiple
    calls (avoids reconnecting on every case).

    Args:
        pathology_summary: Pathology text from PathologyReport.summary.
        base_url:          Optional override for the vLLM endpoint URL.
        model:             Optional override for the model alias.

    Returns:
        TNMResult.
    """
    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["model"] = model

    with StagingSpecialistAgent(**kwargs) as agent:
        return agent.stage(pathology_summary)


# ── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Smoke test for StagingSpecialistAgent."
    )
    p.add_argument(
        "--base_url",
        default=_DEFAULT_BASE_URL,
        help="vLLM OpenAI-compatible base URL.",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="LoRA module alias (--lora-modules name).",
    )
    p.add_argument(
        "--text",
        default=(
            "3.2 cm peripheral lung adenocarcinoma, visceral pleural invasion present, "
            "2/15 lymph nodes positive, no distant metastasis."
        ),
        help="Pathology text to stage.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    agent = StagingSpecialistAgent(base_url=args.base_url, model=args.model)

    print("\n" + "=" * 60)
    print("  StagingSpecialistAgent — smoke test")
    print("=" * 60)
    print(f"  Endpoint : {args.base_url}")
    print(f"  Model    : {args.model}")
    print(f"  Ping     : {'OK' if agent.ping() else 'FAILED (server may be starting)'}")
    print("=" * 60)
    print(f"\nInput:\n  {args.text}\n")

    result = agent.stage(args.text)

    print("Result:")
    print(f"  TNM        : {result.tnm_string()}")
    print(f"  Confidence : {result.confidence}")
    print(f"  Source     : {result.source}")
    print(f"  Latency    : {result.latency_ms} ms")
    if result.error:
        print(f"  Error      : {result.error}")
    print(f"\nFull JSON:\n{json.dumps(result.to_dict(), indent=2)}")
    agent._client.close()
