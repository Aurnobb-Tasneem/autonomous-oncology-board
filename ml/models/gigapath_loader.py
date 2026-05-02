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


# ── Attention Heatmap Extraction ─────────────────────────────────────────────

def _register_attention_hooks(model: nn.Module) -> tuple[list, list]:
    """
    Register forward hooks on all attention blocks to capture attention weights.

    GigaPath is a ViT — each block has an Attention submodule.
    We hook the softmax output (post-dropout attention matrix).

    Returns:
        (hook_handles, attention_maps_list)
    """
    attention_maps: list[torch.Tensor] = []
    handles = []

    def _make_hook():
        def hook(module, input, output):
            # output is the post-attn_drop tensor (B, num_heads, N, N)
            # We save a detached CPU copy to avoid VRAM pressure
            attention_maps.append(output.detach().float().cpu())
        return hook

    for block in model.blocks:
        # timm Attention module: attn_drop is applied after softmax
        # Hook attn_drop to capture the post-softmax attention weights
        handle = block.attn.attn_drop.register_forward_hook(_make_hook())
        handles.append(handle)

    return handles, attention_maps


def _attention_rollout(
    attention_maps: list[torch.Tensor],
    discard_ratio: float = 0.9,
) -> torch.Tensor:
    """
    Attention Rollout (Abnar & Zuidema, 2020) across all transformer layers.

    Recursively multiplies attention matrices through layers to trace
    how information flows from patch tokens to the [CLS] token.

    Args:
        attention_maps: List of (1, num_heads, N, N) tensors, one per block.
        discard_ratio:  Fraction of weakest attentions to zero out per layer.

    Returns:
        Tensor of shape (num_patches,) — normalised attention score per patch.
    """
    num_tokens = attention_maps[0].shape[-1]  # N = 1 + num_patches

    # Start with identity
    result = torch.eye(num_tokens)

    for attn in attention_maps:
        # Average across heads: (N, N)
        attn_avg = attn[0].mean(dim=0)  # (N, N)

        # Add residual connection (identity)
        attn_avg = attn_avg + torch.eye(num_tokens)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

        # Discard low-attention values (keep only top 1-discard_ratio)
        flat = attn_avg.view(-1)
        threshold_val = flat.kthvalue(int(flat.numel() * discard_ratio)).values
        attn_avg[attn_avg < threshold_val] = 0.0

        # Accumulate rollout
        result = attn_avg @ result

    # [CLS] token is index 0 — its attention to all patch tokens
    cls_attention = result[0, 1:]  # shape: (num_patches,)
    cls_attention = (cls_attention - cls_attention.min()) / (
        cls_attention.max() - cls_attention.min() + 1e-8
    )
    return cls_attention


def extract_attention_heatmap(
    model: nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
) -> list[str]:
    """
    Extract GigaPath attention heatmaps for a batch of patches.

    Uses Attention Rollout across all ViT blocks to highlight
    morphologically "suspicious" tissue regions.

    Args:
        model:        GigaPath model (from load_gigapath).
        patch_tensor: Preprocessed patches (N, 3, 224, 224).
        device:       Device the model lives on.

    Returns:
        List of N base64-encoded PNG strings — one heatmap overlay per patch.
        Empty list if extraction fails (graceful degradation).
    """
    import base64
    import io
    import numpy as np

    try:
        import torch.nn.functional as F
        from PIL import Image as PILImage

        # GigaPath patch_size=16 → 14×14 = 196 patches + 1 CLS token
        grid_size = PATCH_SIZE // 16  # = 14

        heatmap_b64s: list[str] = []

        # Process one patch at a time to accumulate per-patch heatmaps
        for idx in range(len(patch_tensor)):
            single = patch_tensor[idx:idx+1].to(device)  # (1, 3, 224, 224)

            # Register hooks
            handles, attn_maps_collected = _register_attention_hooks(model)

            try:
                with torch.no_grad():
                    _ = model(single)
            finally:
                for h in handles:
                    h.remove()

            if not attn_maps_collected:
                # Fallback: return uniform heatmap
                heatmap_b64s.append(_uniform_heatmap_b64())
                continue

            # Attention rollout → (num_patches,) = (196,)
            patch_scores = _attention_rollout(attn_maps_collected)

            # Reshape to grid (14, 14)
            score_grid = patch_scores.reshape(grid_size, grid_size).numpy()

            # Upsample to 224×224
            score_tensor = torch.from_numpy(score_grid).unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
            score_up = F.interpolate(
                score_tensor, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False
            ).squeeze().numpy()  # (224, 224)

            # Convert original patch to RGB for overlay
            orig_rgb = _denormalize_patch(single[0].float().cpu())  # (224, 224, 3) uint8

            # Apply red-to-green colormap
            heatmap_rgb = _apply_colormap(score_up)  # (224, 224, 3) uint8

            # Alpha blend: 60% original + 40% heatmap
            blended = (0.60 * orig_rgb.astype(np.float32) +
                       0.40 * heatmap_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)

            # Add "SUSPICIOUS" label on high-attention patches
            suspicion_level = float(score_up.max())
            if suspicion_level > 0.75:
                blended = _add_label(blended, "⚠ SUSPICIOUS", color=(255, 60, 60))
            elif suspicion_level > 0.5:
                blended = _add_label(blended, "REVIEW", color=(255, 200, 0))

            # Encode to base64 PNG
            img = PILImage.fromarray(blended)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            heatmap_b64s.append(b64)

        return heatmap_b64s

    except Exception as e:
        log.warning(f"GigaPath attention heatmap extraction failed: {e}")
        return []


def _denormalize_patch(tensor: torch.Tensor) -> "np.ndarray":
    """Convert a normalised (3, 224, 224) patch tensor back to uint8 RGB."""
    import numpy as np
    mean = torch.tensor(GIGAPATH_MEAN).view(3, 1, 1)
    std  = torch.tensor(GIGAPATH_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _apply_colormap(score: "np.ndarray") -> "np.ndarray":
    """
    Apply a red-yellow-green colormap to a (H, W) score map in [0, 1].
    High scores (suspicious) → red. Low scores (normal) → green/blue.
    """
    import numpy as np
    # Custom biomedical colormap: blue (0) → yellow (0.5) → red (1)
    r = np.clip(2.0 * score,       0, 1)
    g = np.clip(2.0 * (1 - score), 0, 1) * 0.6
    b = np.clip(1.0 - 2 * score,   0, 1) * 0.8
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype("uint8")


def _add_label(img: "np.ndarray", text: str, color: tuple) -> "np.ndarray":
    """Add a text label to the top-left of an image array."""
    try:
        from PIL import Image as PILImage, ImageDraw
        pil = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Semi-transparent background bar
        draw.rectangle([0, 0, len(text) * 7 + 8, 16], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=color)
        return __import__("numpy").array(pil)
    except Exception:
        return img


def _uniform_heatmap_b64() -> str:
    """Return a neutral grey heatmap as fallback."""
    import base64
    import io
    import numpy as np
    from PIL import Image as PILImage
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Raw Last-Block Attention Scores ──────────────────────────────────────────

def extract_last_block_attention_scores(
    model: nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
) -> list[list[list[float]]]:
    """
    Extract raw CLS attention scores from GigaPath's last transformer block.

    Unlike extract_attention_heatmap() which runs full Attention Rollout across
    all layers and returns base64 PNG overlays, this function returns the raw
    head-averaged attention map from blocks[-1] only.  This is the standard
    DINO-style per-patch saliency score used by frontend ViT visualisation
    libraries and enables interactive WebGL/canvas saliency rendering.

    Architecture note:
        GigaPath ViT-Giant, patch_size=16, input=224×224
        → 14×14 = 196 patch tokens + 1 CLS token = 197 tokens per row/col
        → blocks[-1].attn.attn_drop output: (1, num_heads, 197, 197)
        → CLS row (index 0) → patch columns (indices 1..196) → reshape to 14×14

    Args:
        model:        GigaPath model (from load_gigapath), in eval mode.
        patch_tensor: Preprocessed patches, shape (N, 3, 224, 224).
        device:       Device the model lives on.

    Returns:
        List of N items, each a 14×14 list of floats in [0, 1].
        Outer index = patch index; inner indices = (row, col) of the ViT grid.
        Returns [] on any failure (graceful degradation).

    Example (frontend usage):
        scores = extract_last_block_attention_scores(model, patches, device)
        # scores[0] is a 14×14 grid; scores[0][7][7] is the center token's score
    """
    try:
        import numpy as np

        grid_size = PATCH_SIZE // 16   # = 14 for GigaPath (patch_size=16)
        n_patch_tokens = grid_size * grid_size  # = 196

        all_scores: list[list[list[float]]] = []

        for idx in range(len(patch_tensor)):
            single = patch_tensor[idx : idx + 1].to(device)  # (1, 3, 224, 224)

            # Capture last-block attention only
            last_block_attn: list[torch.Tensor] = []

            def _last_block_hook(module, input, output):
                # output shape: (1, num_heads, 197, 197) — post-softmax, post-dropout
                last_block_attn.append(output.detach().float().cpu())

            handle = model.blocks[-1].attn.attn_drop.register_forward_hook(
                _last_block_hook
            )
            try:
                with torch.no_grad():
                    _ = model(single)
            finally:
                handle.remove()

            if not last_block_attn:
                # Hook didn't fire — architecture mismatch; return uniform map
                all_scores.append(
                    [[1.0 / n_patch_tokens] * grid_size for _ in range(grid_size)]
                )
                continue

            # last_block_attn[0]: (1, num_heads, 197, 197)
            attn = last_block_attn[0][0]          # (num_heads, 197, 197)
            attn_avg = attn.mean(dim=0)            # (197, 197) — head average

            # CLS token (row 0) attending to each patch token (cols 1..196)
            cls_to_patches = attn_avg[0, 1:]       # (196,)

            # Normalise to [0, 1]
            mn, mx = cls_to_patches.min(), cls_to_patches.max()
            if mx - mn > 1e-8:
                cls_to_patches = (cls_to_patches - mn) / (mx - mn)
            else:
                cls_to_patches = torch.zeros_like(cls_to_patches)

            # Reshape to 14×14 grid and convert to nested Python lists (JSON-safe)
            grid = cls_to_patches.reshape(grid_size, grid_size).numpy()
            all_scores.append(
                [[round(float(grid[r, c]), 4) for c in range(grid_size)]
                 for r in range(grid_size)]
            )

        return all_scores

    except Exception as e:
        log.warning(f"GigaPath extract_last_block_attention_scores failed: {e}")
        return []

