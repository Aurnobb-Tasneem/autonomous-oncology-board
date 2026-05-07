"""
scripts/serve_specialists_peft.py
==================================
Drop-in replacement for serve_specialists.sh when vLLM is not available.

Serves the three LoRA specialist adapters via a simple OpenAI-compatible
API on port 8006 using HuggingFace PEFT + Transformers directly.
No vLLM required.

Models served (same aliases as vLLM version):
  - tnm_specialist       → aob/ml/models/checkpoints/tnm_lora
  - biomarker_specialist → aob/ml/models/checkpoints/biomarker_lora
  - treatment_specialist → aob/ml/models/checkpoints/treatment_lora

Usage:
  python scripts/serve_specialists_peft.py

The specialist agents (staging_specialist.py, biomarker_specialist.py,
treatment_specialist.py) will call this server exactly as they would vLLM.
No changes to any agent code required.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = os.getenv("SPECIALIST_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
PORT          = int(os.getenv("SPECIALIST_PORT", "8006"))

# Paths relative to project root (where you run the script from)
ADAPTERS: dict[str, Path] = {
    "tnm_specialist":       Path("aob/ml/models/checkpoints/tnm_lora"),
    "biomarker_specialist": Path("aob/ml/models/checkpoints/biomarker_lora"),
    "treatment_specialist": Path("aob/ml/models/checkpoints/treatment_lora"),
}

# ── Global model state (loaded once, hot-swapped between adapters) ─────────────
_tokenizer   = None
_base_model  = None
_peft_model  = None
_loaded_adapter: Optional[str] = None


def _load_base():
    """Load the base Llama 3.1 8B model once into GPU memory."""
    global _tokenizer, _base_model
    if _base_model is not None:
        return
    log.info(f"Loading base model: {BASE_MODEL_ID}  (this may take 1-2 minutes)")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Detect device — ROCm PyTorch uses "cuda" as device string for AMD GPUs.
    # On some ROCm setups torch.cuda.is_available() may return False even when
    # the GPU is present; fall back to checking torch.version.hip directly.
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.version, "hip", None) is not None:
        device = "cuda"   # ROCm runtime detected — "cuda" is the correct device string
        log.warning("ROCm detected via torch.version.hip but cuda.is_available()=False — "
                    "trying cuda device anyway. Set HIP_VISIBLE_DEVICES=0 if this fails.")
    else:
        device = "cpu"
        log.warning("No GPU detected — running on CPU. Inference will be slow.")
    log.info(f"Loading model onto device: {device}")

    _base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    _base_model.eval()
    log.info(f"Base model loaded onto {device}.")


def _load_adapter(adapter_name: str):
    """Hot-swap the active LoRA adapter."""
    global _peft_model, _loaded_adapter

    if _loaded_adapter == adapter_name and _peft_model is not None:
        return  # Already loaded

    adapter_path = ADAPTERS.get(adapter_name)
    if not adapter_path:
        raise ValueError(f"Unknown adapter: '{adapter_name}'. Valid: {list(ADAPTERS.keys())}")
    if not adapter_path.exists():
        raise ValueError(
            f"Adapter checkpoint not found at '{adapter_path}'. "
            "Run finetune scripts first: python scripts/finetune_tnm.py --max_steps 50"
        )

    _load_base()
    log.info(f"Loading LoRA adapter: {adapter_name} from {adapter_path}")
    _peft_model = PeftModel.from_pretrained(_base_model, str(adapter_path))
    _peft_model.eval()
    _loaded_adapter = adapter_name
    log.info(f"Adapter '{adapter_name}' loaded and ready.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AOB Specialist Server (PEFT — no vLLM required)",
    version="1.0.0",
)


class _Message(BaseModel):
    role: str
    content: str


class _ChatRequest(BaseModel):
    model: str
    messages: List[_Message]
    max_tokens: int = 200
    temperature: float = 0.0
    stop: Optional[List[str]] = None


@app.get("/v1/models")
def list_models():
    """Return all available adapter aliases — mirrors vLLM /v1/models response."""
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "aob-peft-server"}
            for name, path in ADAPTERS.items()
            if path.exists()
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: _ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    t0 = time.perf_counter()

    # Load the requested adapter
    try:
        _load_adapter(req.model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build input using the model's chat template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        input_ids = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(_peft_model.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenisation error: {e}")

    # Generate
    with torch.no_grad():
        output_ids = _peft_model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            do_sample=(req.temperature > 0.01),
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens    = output_ids[0][input_ids.shape[1]:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    latency_ms    = round((time.perf_counter() - t0) * 1000)

    log.info(f"[{req.model}] {latency_ms}ms → {response_text[:120]}")

    return {
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     int(input_ids.shape[1]),
            "completion_tokens": int(len(new_tokens)),
            "total_tokens":      int(input_ids.shape[1]) + int(len(new_tokens)),
        },
    }


@app.on_event("startup")
async def startup_event():
    log.info("=" * 60)
    log.info("  AOB Specialist Server — PEFT mode (no vLLM)")
    log.info(f"  Base model : {BASE_MODEL_ID}")
    log.info(f"  Port       : {PORT}")
    for name, path in ADAPTERS.items():
        status = "✅ found" if path.exists() else "❌ missing"
        log.info(f"  {name:<25} {status}  ({path})")
    log.info("=" * 60)
    # Pre-load base model at startup so first request is fast
    _load_base()
    log.info("✅ Server ready — waiting for requests on port 8006")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
