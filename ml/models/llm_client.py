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

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_HOST  = os.getenv("OLLAMA_HOST", "http://172.17.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:70b")


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
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
    ):
        self.host    = host.rstrip("/")
        self.model   = model
        self.timeout = timeout
        log.info(f"OllamaClient: host={self.host}  model={self.model}")

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
