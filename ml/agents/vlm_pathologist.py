"""
ml/agents/vlm_pathologist.py
=============================
Agent 1b: VLM Pathologist — Qwen2.5-VL-7B-Instruct visual second opinion.

Track 3 (Vision & Multimodal AI) + Qwen Challenge deliverable.

What this agent does
---------------------
Takes raw LC25000 image patches (PIL.Image), feeds them directly into
Qwen/Qwen2.5-VL-7B-Instruct, and asks for clinical morphological descriptions.
This is a pixel-level visual analysis that is entirely independent of
GigaPath's embedding-based classification — a genuine second opinion.

The output (VLMOpinion) is passed to MetaEvaluator.reconcile() which
compares it with the GigaPath PathologyReport to detect agreements and
discrepancies before the Oncologist drafts the management plan.

Architecture
-------------
                  ┌─────────────────────────────────┐
  Image patches   │   Qwen2.5-VL-7B-Instruct         │
  (PIL.Image) ──▶ │   AutoModelForVision2Seq          │──▶ VLMOpinion
                  │   ~15 GB BF16 on AMD MI300X       │
                  └─────────────────────────────────────┘
                  ┌─────────────────────────────────┐
  VLMOpinion  ──▶ │   MetaEvaluator.reconcile()      │──▶ agreement_score
  + GigaPath   │   Llama 3.3 70B                   │    + discrepancies
  Report       └─────────────────────────────────────┘

VRAM budget (AMD MI300X, 192 GB HBM3)
---------------------------------------
  GigaPath (FP16, already resident)    ~  3 GB
  Llama 3.3 70B via Ollama             ~ 40 GB
  Qwen2.5-VL-7B-Instruct (BF16, new)  ~ 15 GB
  ────────────────────────────────────────────
  Total                                ~ 58 GB   (134 GB headroom)

Usage
-----
    from ml.agents.vlm_pathologist import VLMPathologistAgent
    from PIL import Image

    agent = VLMPathologistAgent(hf_token="hf_...")
    images = [Image.open("patch.jpg")]
    opinion = agent.describe(images)
    print(opinion.aggregate_description)
    # → "Glandular structures with nuclear atypia and irregular borders..."
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Optional

from PIL import Image

log = logging.getLogger(__name__)

# ── Model identifier ─────────────────────────────────────────────────────────
QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Prompts ───────────────────────────────────────────────────────────────────
_MORPHOLOGY_PROMPT = (
    "You are a board-certified digital pathologist reviewing a histopathology image patch. "
    "Describe the key morphological features you observe. "
    "Focus on: cell architecture, nuclear characteristics (size, shape, chromatin), "
    "stromal patterns, mitotic figures, and any features suggesting malignancy. "
    "Be concise and precise (2–3 sentences)."
)

_TISSUE_TYPE_PROMPT = (
    "Based on this pathology description:\n\n"
    '"{description}"\n\n'
    "What is the most likely tissue type / cancer diagnosis? "
    "Reply with only the tissue type name (e.g. 'lung adenocarcinoma', "
    "'colon adenocarcinoma', 'lung squamous cell carcinoma', 'benign tissue')."
)

_MALIGNANCY_PROMPT = (
    "Based on this pathology description:\n\n"
    '"{description}"\n\n'
    "List up to 5 key morphological indicators of malignancy you can infer. "
    "Reply with a JSON array of short strings, e.g. "
    '["nuclear atypia", "increased mitotic figures"]. '
    "If no malignancy indicators are present, reply with []."
)

# Max patches to process — keeps latency under ~10s on MI300X
DEFAULT_MAX_PATCHES = 4

# Generation params
_GENERATE_KWARGS = {
    "max_new_tokens":  120,
    "do_sample":       False,   # greedy — reproducible morphology descriptions
    "temperature":     1.0,     # unused when do_sample=False but avoids vLLM warnings
}


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class VLMOpinion:
    """
    Visual morphology opinion from Qwen2.5-VL-7B-Instruct.

    Produced by VLMPathologistAgent.describe() and consumed by
    MetaEvaluator.reconcile() to cross-check GigaPath's findings.
    """
    per_patch_descriptions: list[str]   # one morphology description per patch
    aggregate_description:  str         # concatenated summary across all patches
    n_patches_processed:    int
    suspected_tissue_type:  str         # extracted by text-only follow-up prompt
    malignancy_indicators:  list[str]   # e.g. ["nuclear atypia", "gland irregularity"]
    processing_time_s:      float
    model_id:               str         # always "Qwen/Qwen2.5-VL-7B-Instruct"
    error:                  Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_available(self) -> bool:
        return self.error is None and bool(self.aggregate_description)


def _empty_opinion(error: str, elapsed: float = 0.0) -> VLMOpinion:
    """Return a graceful fallback VLMOpinion when the model is unavailable."""
    return VLMOpinion(
        per_patch_descriptions=[],
        aggregate_description="",
        n_patches_processed=0,
        suspected_tissue_type="unavailable",
        malignancy_indicators=[],
        processing_time_s=elapsed,
        model_id=QWEN_VL_MODEL_ID,
        error=error,
    )


# ── Model loading (cached) ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_qwen_vl(hf_token: str, device_str: str):
    """
    Load Qwen2.5-VL-7B-Instruct model and processor (cached — loads once per process).

    Args:
        hf_token:   HuggingFace access token (can be "" for public models).
        device_str: "cuda" or "cpu" (string — lru_cache needs hashable args).

    Returns:
        (model, processor) tuple. Model is in eval mode, bf16 on CUDA.
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from huggingface_hub import login as hf_login

    if hf_token:
        hf_login(token=hf_token, add_to_git_credential=False)
    log.info("Qwen2.5-VL: HF token authenticated")

    dtype = torch.bfloat16 if device_str == "cuda" else torch.float32
    log.info(f"Qwen2.5-VL: loading {QWEN_VL_MODEL_ID} on {device_str} ({dtype}) ...")

    processor = AutoProcessor.from_pretrained(
        QWEN_VL_MODEL_ID,
        token=hf_token or None,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_VL_MODEL_ID,
        token=hf_token or None,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"Qwen2.5-VL: loaded {n_params:.1f}B parameters on {device_str}")
    return model, processor


# ── Agent ────────────────────────────────────────────────────────────────────

class VLMPathologistAgent:
    """
    Agent 1b: Visual Language Model Pathologist.

    Uses Qwen2.5-VL-7B-Instruct to produce clinical morphological descriptions
    of histopathology image patches. Runs independently of GigaPath — its
    findings are reconciled by MetaEvaluator to strengthen the board consensus.

    Args:
        hf_token:    HuggingFace token. Reads HF_TOKEN env var if not set.
                     Qwen2-VL-7B is public so token is optional but recommended.
        device:      "cuda" (auto-selects ROCm on MI300X), "cpu", or None (auto).
        max_patches: Maximum patches to describe per call (default 4).
    """

    def __init__(
        self,
        hf_token:    Optional[str] = None,
        device:      Optional[str] = None,
        max_patches: int = DEFAULT_MAX_PATCHES,
    ):
        self._hf_token   = hf_token or os.getenv("HF_TOKEN", "")
        self._device_str: Optional[str] = device
        self._max_patches = max_patches
        self._model      = None
        self._processor  = None
        log.info("VLMPathologistAgent: initialised (model load deferred to first call)")

    def _ensure_loaded(self):
        """Lazy model loading — happens only on first describe() call."""
        if self._model is not None:
            return

        import torch
        if self._device_str is None:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self._model, self._processor = _load_qwen_vl(
            self._hf_token, self._device_str
        )

    # ── Per-patch description ─────────────────────────────────────────────────

    def _describe_patch(self, image: Image.Image) -> str:
        """
        Generate a morphological description for one image patch.

        Uses Qwen2-VL's native multimodal message format with an embedded PIL
        image. qwen_vl_utils.process_vision_info handles the image tensor
        preparation (resize, normalise) according to the model's requirements.

        Args:
            image: PIL.Image patch (any size — processor resizes internally).

        Returns:
            Morphology description string.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type":  "image",
                        "image": image,   # PIL.Image accepted directly
                    },
                    {
                        "type": "text",
                        "text": _MORPHOLOGY_PROMPT,
                    },
                ],
            }
        ]

        # Apply Qwen2-VL chat template (adds special vision tokens)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                **_GENERATE_KWARGS,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude input)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        description = self._processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return description

    # ── Text-only follow-up prompts ───────────────────────────────────────────

    def _extract_tissue_type(self, aggregate_description: str) -> str:
        """
        Ask Qwen2-VL (text-only mode) to identify the likely tissue type.

        No image — just a text completion over the aggregate description.
        This mirrors how a pathologist reads a colleague's written report.
        """
        import torch

        prompt = _TISSUE_TYPE_PROMPT.format(description=aggregate_description[:500])
        messages = [{"role": "user", "content": prompt}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        tissue_type = self._processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return tissue_type or "unspecified"

    def _extract_malignancy_indicators(self, aggregate_description: str) -> list[str]:
        """
        Ask Qwen2-VL (text-only) to list malignancy indicators as JSON array.
        Returns a list of short strings, or [] if none found.
        """
        import json
        import re
        import torch

        prompt = _MALIGNANCY_PROMPT.format(description=aggregate_description[:500])
        messages = [{"role": "user", "content": prompt}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Extract JSON array from the response
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            try:
                indicators = json.loads(m.group(0))
                if isinstance(indicators, list):
                    return [str(x).strip() for x in indicators[:5]]
            except json.JSONDecodeError:
                pass

        # Fallback: split by comma or newline if JSON failed
        if raw and raw != "[]":
            parts = [p.strip().strip('"').strip("'") for p in re.split(r"[,\n]", raw)]
            return [p for p in parts if len(p) > 3][:5]

        return []

    # ── Main method ───────────────────────────────────────────────────────────

    def describe(
        self,
        images: list[Image.Image],
        max_patches: Optional[int] = None,
    ) -> VLMOpinion:
        """
        Generate visual morphology descriptions for a set of image patches.

        Args:
            images:      List of PIL.Image patches from the pathology case.
            max_patches: Cap on patches to process (overrides instance default).
                         Recommended: 4 patches for <10s latency on MI300X.

        Returns:
            VLMOpinion with per-patch descriptions, aggregate summary,
            suspected tissue type, and malignancy indicators.
            On model error, returns VLMOpinion with error field set.
        """
        t0 = time.perf_counter()
        cap = max_patches if max_patches is not None else self._max_patches

        if not images:
            return _empty_opinion("No images provided.")

        try:
            self._ensure_loaded()
        except Exception as load_err:
            elapsed = round(time.perf_counter() - t0, 2)
            msg = f"Qwen2.5-VL model load failed: {load_err}"
            log.warning(f"VLMPathologistAgent: {msg}")
            return _empty_opinion(msg, elapsed)

        patches = images[:cap]
        log.info(
            f"VLMPathologistAgent: describing {len(patches)}/{len(images)} patches "
            f"with {QWEN_VL_MODEL_ID}"
        )

        per_patch: list[str] = []
        for idx, patch in enumerate(patches):
            try:
                desc = self._describe_patch(patch.convert("RGB"))
                per_patch.append(desc)
                log.debug(f"VLMPathologistAgent: patch {idx}: {desc[:80]}...")
            except Exception as e:
                log.warning(f"VLMPathologistAgent: patch {idx} failed — {e}")
                per_patch.append("")

        # Filter empty descriptions from failed patches
        valid_descriptions = [d for d in per_patch if d]
        if not valid_descriptions:
            elapsed = round(time.perf_counter() - t0, 2)
            return _empty_opinion("All patch descriptions failed.", elapsed)

        # Aggregate: join with separator so downstream models see a coherent text
        aggregate = " | ".join(valid_descriptions)

        # Follow-up text prompts (no images) to extract structured fields
        try:
            tissue_type = self._extract_tissue_type(aggregate)
        except Exception as e:
            log.warning(f"VLMPathologistAgent: tissue type extraction failed — {e}")
            tissue_type = "unspecified"

        try:
            malignancy_indicators = self._extract_malignancy_indicators(aggregate)
        except Exception as e:
            log.warning(f"VLMPathologistAgent: malignancy extraction failed — {e}")
            malignancy_indicators = []

        elapsed = round(time.perf_counter() - t0, 2)

        log.info(
            f"VLMPathologistAgent: done in {elapsed}s — "
            f"tissue='{tissue_type}'  indicators={malignancy_indicators[:3]}"
        )

        return VLMOpinion(
            per_patch_descriptions=per_patch,
            aggregate_description=aggregate,
            n_patches_processed=len(valid_descriptions),
            suspected_tissue_type=tissue_type,
            malignancy_indicators=malignancy_indicators,
            processing_time_s=elapsed,
            model_id=QWEN_VL_MODEL_ID,
            error=None,
        )
