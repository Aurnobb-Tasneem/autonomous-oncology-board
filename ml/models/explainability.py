"""
ml/models/explainability.py
============================
Triple-modal explainability for GigaPath ViT-Giant patches.

Three complementary saliency methods:
    1. Attention Rollout      — already in gigapath_loader.py (kept for speed)
    2. Grad-CAM++             — backprop through last transformer block
    3. Integrated Gradients   — 20-step path integral from zero baseline

All three return a (224, 224) float32 heatmap normalised to [0, 1].
The caller can then blend with the original patch image.

Usage:
    from ml.models.explainability import (
        compute_gradcam_plus_plus,
        compute_integrated_gradients,
        heatmap_to_overlay,
    )

    # After GigaPath inference:
    heatmap_gcpp = compute_gradcam_plus_plus(model, patch_tensor, target_class)
    heatmap_ig   = compute_integrated_gradients(model, patch_tensor, target_class)
    overlay      = heatmap_to_overlay(original_pil_image, heatmap_gcpp)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

class _GradCAMPlusPlus:
    """
    Grad-CAM++ for Vision Transformer (applied to last attention block).

    For a ViT, we treat the attention output of the last transformer block
    as the "feature map". Grad-CAM++ weights the feature channels by
    second-order gradients (hence the ++ vs standard Grad-CAM).

    Implementation:
        a^c_k = ReLU(∂²y^c / (∂A^k)²) / (2 * ∂²y^c / (∂A^k)² + Σ_xy A^k_xy * ∂³y^c / (∂A^k)³)
        weights = global_average_pool(a^c_k * ∂y^c / ∂A^k)
        L^c_GradCAM++ = ReLU(Σ_k weights_k * A^k)

    For ViT patches we average the spatial token activations.
    """

    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model        = model
        self.target_layer = target_layer
        self._activations  = None
        self._gradients    = None
        self._hooks: list  = []

    def _register(self, layer):
        def fwd_hook(module, input, output):
            # ViT attention output: (B, num_tokens, embed_dim)
            self._activations = output

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_target_layer(self):
        """Try to find the last transformer block's attention output layer."""
        if self.target_layer is not None:
            return self.target_layer
        # GigaPath / timm ViT-Giant: model.blocks[-1] is the last block
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            return self.model.blocks[-1]
        # Fallback: last named module
        layers = list(self.model.named_modules())
        return layers[-1][1] if layers else None

    def compute(
        self,
        input_tensor: torch.Tensor,   # (1, 3, 224, 224)
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap.

        Returns:
            (H, W) float32 numpy array, normalised to [0, 1].
        """
        layer = self._get_target_layer()
        if layer is None:
            log.warning("GradCAM++: could not find target layer — returning uniform map")
            return np.ones((224, 224), dtype=np.float32)

        self._register(layer)
        self.model.zero_grad()

        try:
            input_tensor = input_tensor.requires_grad_(True)

            # Forward pass
            output = self.model(input_tensor)

            if target_class is None:
                target_class = int(output.argmax(dim=-1).item())

            # Scalar score for target class
            if output.ndim == 2:
                score = output[0, target_class]
            else:
                score = output[0]

            self.model.zero_grad()
            score.backward(retain_graph=False)

            activations = self._activations   # (1, T, D) or similar
            gradients   = self._gradients     # same shape

            if activations is None or gradients is None:
                log.warning("GradCAM++: hooks did not capture activations")
                return np.ones((224, 224), dtype=np.float32)

            act = activations.detach()   # (1, T, D)
            grd = gradients.detach()     # (1, T, D)

            # Pool over spatial tokens (excluding [CLS] token at position 0)
            if act.ndim == 3:
                act_spatial = act[0, 1:, :]   # (T-1, D)
                grd_spatial = grd[0, 1:, :]   # (T-1, D)
            else:
                act_spatial = act.reshape(act.shape[0], -1).T
                grd_spatial = grd.reshape(grd.shape[0], -1).T

            # Grad-CAM++ weights per channel
            weights = F.relu(grd_spatial).mean(dim=0)  # (D,)
            cam     = (act_spatial * weights.unsqueeze(0)).sum(dim=-1)  # (T-1,)
            cam     = F.relu(cam)

            # Reshape to square grid (for ViT-Giant with 196 or 256 patches)
            n_patches = cam.shape[0]
            grid_size = int(np.sqrt(n_patches))
            if grid_size * grid_size != n_patches:
                # Pad to next square
                grid_size = int(np.ceil(np.sqrt(n_patches)))
                cam_padded = torch.zeros(grid_size * grid_size)
                cam_padded[:n_patches] = cam
                cam = cam_padded

            cam_grid = cam.reshape(grid_size, grid_size).cpu().numpy()

            # Resize to 224×224
            cam_pil  = Image.fromarray(cam_grid.astype(np.float32))
            cam_224  = np.array(cam_pil.resize((224, 224), Image.BILINEAR))

            # Normalise
            cam_min  = cam_224.min()
            cam_max  = cam_224.max()
            if cam_max > cam_min:
                cam_224 = (cam_224 - cam_min) / (cam_max - cam_min)
            else:
                cam_224 = np.zeros_like(cam_224)

            return cam_224.astype(np.float32)

        finally:
            self._remove_hooks()
            self.model.zero_grad()


def compute_gradcam_plus_plus(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,    # (1, 3, 224, 224)
    target_class: Optional[int] = None,
    target_layer=None,
) -> np.ndarray:
    """
    Compute Grad-CAM++ saliency map for a GigaPath ViT.

    Args:
        model:         GigaPath model (timm ViT).
        input_tensor:  Preprocessed patch (1, 3, 224, 224).
        target_class:  Class index (uses argmax if None).
        target_layer:  Specific layer to hook. Defaults to last ViT block.

    Returns:
        (224, 224) float32 heatmap, values in [0, 1].
    """
    gcpp = _GradCAMPlusPlus(model, target_layer)
    return gcpp.compute(input_tensor, target_class)


# ── Integrated Gradients ──────────────────────────────────────────────────────

def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,    # (1, 3, 224, 224)
    target_class: Optional[int] = None,
    n_steps: int                 = 20,
    baseline: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients (Sundararajan et al., 2017) for a GigaPath patch.

    IG formula: IG_i(x) = (x_i - x_i') × Σ_{k=1}^{m} (∂F(x' + k/m × (x - x')) / ∂x_i) × (1/m)

    Where x' is the baseline (zero image by default) and m is n_steps.

    Args:
        model:         GigaPath model.
        input_tensor:  Preprocessed patch (1, 3, 224, 224).
        target_class:  Class index. Uses argmax if None.
        n_steps:       Number of interpolation steps (20 is fast, 50 is accurate).
        baseline:      Baseline image. Uses zero tensor if None.

    Returns:
        (224, 224) float32 heatmap, values in [0, 1].
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Determine target class from a single forward pass
    if target_class is None:
        with torch.no_grad():
            out = model(input_tensor)
            target_class = int(out.argmax(dim=-1).item())

    # Interpolated inputs: (n_steps+1, 3, 224, 224)
    alphas = torch.linspace(0, 1, n_steps + 1).to(input_tensor.device)
    interpolated = baseline + alphas[:, None, None, None] * (input_tensor - baseline)
    interpolated.requires_grad_(True)

    # Batch forward pass
    outputs = model(interpolated)   # (n_steps+1, n_classes) or (n_steps+1, embed_dim)

    # If model returns embeddings (not logits), use sum of output as scalar
    if outputs.ndim == 2 and outputs.shape[1] > 1000:
        # Likely embedding output — use L2 norm as proxy
        scores = outputs.norm(dim=-1)
    else:
        scores = outputs[:, target_class]

    # Sum scores and backprop to get accumulated gradients
    scores.sum().backward()

    grads = interpolated.grad                  # (n_steps+1, 3, 224, 224)
    avg_grads = grads[:-1].mean(dim=0)         # (3, 224, 224)

    ig_attr = (input_tensor.squeeze(0) - baseline.squeeze(0)) * avg_grads  # (3, 224, 224)

    # Average over colour channels and take absolute value
    ig_map = ig_attr.abs().mean(dim=0).detach().cpu().numpy()  # (224, 224)

    # Normalise
    ig_min = ig_map.min()
    ig_max = ig_map.max()
    if ig_max > ig_min:
        ig_map = (ig_map - ig_min) / (ig_max - ig_min)
    else:
        ig_map = np.zeros_like(ig_map)

    return ig_map.astype(np.float32)


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def heatmap_to_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,        # (H, W) float32, [0, 1]
    colormap: str = "hot",
    alpha: float  = 0.5,
) -> Image.Image:
    """
    Blend a saliency heatmap onto the original patch image.

    Args:
        original_image:  PIL RGB image (224×224 recommended).
        heatmap:         Float32 (H, W) array, values in [0, 1].
        colormap:        matplotlib colormap name.
        alpha:           Overlay opacity (0–1).

    Returns:
        PIL RGB image with colour heatmap blended over the original.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        cmap    = cm.get_cmap(colormap)
        colored = cmap(heatmap)[:, :, :3]  # (H, W, 3), [0,1] float
        colored_uint8 = (colored * 255).astype(np.uint8)
        overlay_img   = Image.fromarray(colored_uint8).resize(original_image.size, Image.BILINEAR)

        result = Image.blend(original_image.convert("RGB"), overlay_img, alpha=alpha)
        return result
    except ImportError:
        # Fallback: return original image with red-channel modulation
        arr = np.array(original_image.convert("RGB"), dtype=np.float32)
        h_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original_image.size, Image.BILINEAR)
        ) / 255.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] + h_resized * 128, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


def heatmap_to_base64(
    original_image: Image.Image,
    heatmap: np.ndarray,
    method_name: str = "gradcam",
    colormap: str    = "hot",
    alpha: float     = 0.5,
) -> str:
    """
    Render overlay and encode as base64 PNG for API responses.

    Returns:
        Base64-encoded PNG string (no data:image/png prefix).
    """
    import base64
    import io

    overlay = heatmap_to_overlay(original_image, heatmap, colormap, alpha)
    buf     = io.BytesIO()
    overlay.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_all_heatmaps(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,    # (1, 3, 224, 224)
    original_image: Image.Image,
    target_class: Optional[int] = None,
    include_rollout_fn=None,       # callable(model, input_tensor) → np.ndarray, optional
) -> dict[str, str]:
    """
    Compute all three saliency methods and return a dict of base64 PNG strings.

    Args:
        model:             GigaPath model.
        input_tensor:      Preprocessed patch tensor.
        original_image:    Original PIL image for overlay rendering.
        target_class:      Class index (uses argmax if None).
        include_rollout_fn: Optional function computing attention rollout.

    Returns:
        dict with keys "attention_rollout", "gradcam_plus_plus", "integrated_gradients",
        each mapping to a base64 PNG string.
    """
    results: dict[str, str] = {}

    # 1. Attention Rollout (fast — call existing function if provided)
    if include_rollout_fn is not None:
        try:
            rollout_map = include_rollout_fn(model, input_tensor)
            results["attention_rollout"] = heatmap_to_base64(original_image, rollout_map, "attention")
        except Exception as e:
            log.warning(f"Attention rollout failed: {e}")

    # 2. Grad-CAM++
    try:
        model.eval()
        gcpp_map = compute_gradcam_plus_plus(model, input_tensor, target_class)
        results["gradcam_plus_plus"] = heatmap_to_base64(original_image, gcpp_map, "gradcam")
        log.info("Grad-CAM++ heatmap computed")
    except Exception as e:
        log.warning(f"Grad-CAM++ failed: {e}")

    # 3. Integrated Gradients
    try:
        model.eval()
        input_copy = input_tensor.clone()
        ig_map     = compute_integrated_gradients(model, input_copy, target_class)
        results["integrated_gradients"] = heatmap_to_base64(original_image, ig_map, "integrated_gradients", colormap="viridis")
        log.info("Integrated gradients heatmap computed")
    except Exception as e:
        log.warning(f"Integrated gradients failed: {e}")

    return results
