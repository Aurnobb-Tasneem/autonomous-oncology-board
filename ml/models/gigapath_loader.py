"""
ml/models/gigapath_loader.py
============================
GigaPath model loading and patch preprocessing utilities.

GigaPath is a ViT-Giant pre-trained on 1.3B pathology image tokens.
It outputs 1536-dimensional embeddings per 224x224 patch.

HuggingFace: prov-gigapath/prov-gigapath
Paper: https://www.nature.com/articles/s41586-024-07441-w
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms

log = logging.getLogger(__name__)

# ── GigaPath preprocessing constants ────────────────────────────────────────
# From the GigaPath paper — same normalisation used during pre-training.
GIGAPATH_MEAN = (0.485, 0.456, 0.406)
GIGAPATH_STD  = (0.229, 0.224, 0.225)
PATCH_SIZE     = 224   # pixels — GigaPath is trained on 224×224 patches
EMBEDDING_DIM  = 1536  # ViT-Giant output dimension


def build_transform(augment: bool = False) -> transforms.Compose:
    """
    Build the image preprocessing pipeline for GigaPath.

    Args:
        augment: If True, add random flip/colour jitter for data augmentation.
                 Only use during calibration, never during inference.
    """
    base = [
        transforms.Resize(PATCH_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=GIGAPATH_MEAN, std=GIGAPATH_STD),
    ]
    if augment:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ] + base
    return transforms.Compose(base)


@lru_cache(maxsize=1)
def load_gigapath(
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, torch.device]:
    """
    Load Prov-GigaPath from HuggingFace (cached — loads only once per process).

    Args:
        hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
        device:   'cuda', 'cpu', or None (auto-select).
        dtype:    Model dtype. FP16 on GPU, FP32 on CPU.

    Returns:
        (model, device) tuple. Model is in eval mode.

    Raises:
        RuntimeError: If HF token is missing or model access is not approved.
    """
    import timm
    from huggingface_hub import login as hf_login

    token = hf_token or os.getenv("HF_TOKEN", "")
    if token:
        hf_login(token=token, add_to_git_credential=False)
        log.info("GigaPath: authenticated with HF token")
    else:
        log.warning("GigaPath: no HF_TOKEN — anonymous access (may fail for gated model)")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cpu":
        dtype = torch.float32  # FP16 unsupported on CPU

    log.info(f"GigaPath: loading model on {device} ({dtype}) ...")
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model = model.eval().to(device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"GigaPath: loaded {n_params:.0f}M parameters")
    return model, device


@torch.inference_mode()
def embed_patches(
    model: nn.Module,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Extract GigaPath embeddings for a batch of patches.

    Args:
        model:      GigaPath model (from load_gigapath).
        patches:    Tensor of shape (N, 3, 224, 224) in the model's dtype.
        device:     Device the model lives on.
        batch_size: Mini-batch size for inference (tune to VRAM).

    Returns:
        Tensor of shape (N, 1536) on CPU — one embedding per patch.
    """
    embeddings = []
    patches = patches.to(device)

    for i in range(0, len(patches), batch_size):
        chunk = patches[i : i + batch_size]
        emb = model(chunk)   # shape: (chunk_size, 1536)
        embeddings.append(emb.float().cpu())

    return torch.cat(embeddings, dim=0)   # (N, 1536)
