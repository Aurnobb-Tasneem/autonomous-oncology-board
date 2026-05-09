"""
ml/models/llm_client.py
=======================
Ollama LLM client wrapper — replaces vLLM for ROCm compatibility.

Ollama is ROCm-native and serves Llama 3.3 70B on the AMD MI300X.
It exposes an OpenAI-compatible REST API at port 11434.

Usage:
    client = OllamaClient()
    response = await client.generate("Summarise this pathology report: ...")
    # or sync:
    response = client.generate_sync("...")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

def _default_ollama_host() -> str:
    v = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return v.strip() if v and v.strip() else "http://localhost:11434"


def _default_ollama_model() -> str:
    v = os.getenv("OLLAMA_MODEL", "llama3.3:70b-instruct-q4_K_S")
    return v.strip() if v and v.strip() else "llama3.3:70b-instruct-q4_K_S"


def _default_ollama_timeout_s() -> int:
    """HTTP read timeout for /api/generate and /api/chat (70B + long prompts need headroom)."""
    raw = os.getenv("OLLAMA_REQUEST_TIMEOUT", "600")
    try:
        return max(30, int(raw))
    except ValueError:
        return 600


def _ollama_keep_alive() -> str:
    """Keep model loaded between board steps (Ollama duration string, e.g. 30m)."""
    v = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
    return v if v else "30m"


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_HOST  = _default_ollama_host()
DEFAULT_MODEL = _default_ollama_model()


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ms: float


class OllamaClient:
    """
    Synchronous Ollama client for the AOB backend.

    The oncologist agent uses this to generate the final Patient Management Plan.
    The researcher agent uses it to synthesise RAG evidence into citations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        # Treat None / blank like “unset” so explicit host=None never overrides DEFAULT_*.
        _h = (host or "").strip() or DEFAULT_HOST
        _m = (model or "").strip() or DEFAULT_MODEL
        self.host    = _h.rstrip("/")
        self.model   = _m
        self.timeout = timeout if timeout is not None else _default_ollama_timeout_s()
        log.info(f"OllamaClient: host={self.host}  model={self.model}  timeout={self.timeout}s")

    # ── Health check ────────────────────────────────────────────────────────
    def ping(self) -> bool:
        """Return True if Ollama is reachable and model is available."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                available = any(self.model.split(":")[0] in m for m in models)
                if not available:
                    log.warning(f"OllamaClient: model '{self.model}' not found. Pull it first.")
                return available
        except Exception as e:
            log.error(f"OllamaClient: ping failed — {e}")
            return False

    # ── Core generate ───────────────────────────────────────────────────────
    def generate_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Synchronous text generation via Ollama /api/generate.

        Args:
            prompt:      The user prompt.
            system:      Optional system prompt (role definition).
            temperature: Sampling temperature (0 = deterministic).
            max_tokens:  Maximum tokens to generate.

        Returns:
            LLMResponse dataclass with generated text and token counts.
        """
        payload: dict = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": _ollama_keep_alive(),
            "options": {
                "temperature":  temperature,
                "num_predict":  max_tokens,
                "num_ctx":      4096,
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        return LLMResponse(
            text=result.get("response", "").strip(),
            model=result.get("model", self.model),
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_duration_ms=result.get("total_duration", 0) / 1_000_000,
        )

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Back-compat helper for agents that expect OpenAI-style kwargs.

        Returns raw text only (not LLMResponse). Same backend as generate_sync.
        """
        r = self.generate_sync(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return r.text

    def generate_with_context(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Chat completion via Ollama /api/chat (OpenAI-compatible messages format).

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated text.
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "keep_alive": _ollama_keep_alive(),
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx":     4096,
            },
        }
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama chat request failed: {e}") from e

        message = result.get("message", {})
        return LLMResponse(
            text=message.get("content", "").strip(),
            model=result.get("model", self.model),
            prompt_tokens=result.get("prompt_eval_count", 0),
            completion_tokens=result.get("eval_count", 0),
            total_duration_ms=result.get("total_duration", 0) / 1_000_000,
        )
