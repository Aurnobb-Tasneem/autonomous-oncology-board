"""
ml/training/giga_head.py
=========================
GigaPath MLP classification head training on LC25000.

Replaces prototype-based cosine similarity with a real trained 2-layer MLP:
    GigaPath embedding (1536-d, frozen) → MLP head → 5-class softmax

Architecture:
    Linear(1536 → 512) + BatchNorm + ReLU + Dropout(0.2)
    → Linear(512 → 5)

Training:
    - Freeze GigaPath weights entirely
    - Train only the MLP head (~800K params)
    - LC25000: 5 classes × 5000 images = 25k total
    - ~10 min on MI300X at batch=128

Usage:
    python -m ml.training.giga_head \
        --data_root /path/to/LC25000 \
        --output_dir aob/ml/models/checkpoints/giga_head \
        --epochs 10

    # Smoke test (no GPU required):
    python -m ml.training.giga_head --max_samples 100 --epochs 2

The trained head is loaded by PathologistAgent instead of prototype matching.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

TISSUE_CLASSES = [
    "colon_adenocarcinoma",
    "colon_benign_tissue",
    "lung_adenocarcinoma",
    "lung_benign_tissue",
    "lung_squamous_cell_carcinoma",
]
EMBEDDING_DIM = 1536
N_CLASSES     = len(TISSUE_CLASSES)


# ── Model ─────────────────────────────────────────────────────────────────────

class GigaPathHead(nn.Module):
    """
    2-layer MLP classification head for GigaPath embeddings.

    Input:  1536-d GigaPath embedding (already L2-normalised by encoder)
    Output: 5-class logits
    """

    def __init__(
        self,
        in_dim: int    = EMBEDDING_DIM,
        hidden_dim: int = 512,
        n_classes: int  = N_CLASSES,
        dropout: float  = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class_index, probabilities)."""
        proba = self.predict_proba(x)
        return proba.argmax(dim=-1), proba


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Holds pre-computed GigaPath embeddings + class labels.
    Supports both in-memory tensors and on-disk .pt files.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,   # (N, 1536)
        labels: torch.Tensor,       # (N,) long
    ):
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings
        self.labels     = labels

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings_from_folder(
    data_root: Path,
    gigapath_model,
    transform,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Walk data_root looking for class subdirectories matching TISSUE_CLASSES.
    Extract GigaPath embeddings for all images (or max_samples if set).

    Expected directory structure:
        data_root/
            colon_adenocarcinoma/    ← class name matches TISSUE_CLASSES
            colon_benign_tissue/
            lung_adenocarcinoma/
            lung_benign_tissue/
            lung_squamous_cell_carcinoma/

    Returns:
        embeddings: (N, 1536) float32 tensor
        labels:     (N,)      long tensor
    """
    from PIL import Image

    if device is None:
        device = next(gigapath_model.parameters()).device

    all_embeddings = []
    all_labels     = []

    gigapath_model.eval()

    for class_idx, class_name in enumerate(TISSUE_CLASSES):
        class_dir = data_root / class_name
        if not class_dir.exists():
            # Try case-insensitive match
            matches = [d for d in data_root.iterdir() if d.name.lower() == class_name.lower()]
            if matches:
                class_dir = matches[0]
            else:
                log.warning(f"Class dir not found: {class_dir} — skipping")
                continue

        image_paths = sorted(
            list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        )

        if max_samples is not None:
            per_class = max(1, max_samples // N_CLASSES)
            image_paths = image_paths[:per_class]

        log.info(f"  [{class_name}] {len(image_paths)} images")

        # Process in batches
        imgs_batch = []
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                imgs_batch.append(transform(img))
            except Exception as e:
                log.warning(f"  Skip {img_path.name}: {e}")
                continue

            if len(imgs_batch) == batch_size or i == len(image_paths) - 1:
                if not imgs_batch:
                    continue
                batch_tensor = torch.stack(imgs_batch).to(device)
                with torch.no_grad():
                    emb = gigapath_model(batch_tensor)  # (B, 1536)
                    emb = F.normalize(emb, dim=-1)
                all_embeddings.append(emb.cpu().float())
                all_labels.extend([class_idx] * emb.shape[0])
                imgs_batch = []

    if not all_embeddings:
        raise RuntimeError(f"No images found under {data_root}")

    return (
        torch.cat(all_embeddings, dim=0),
        torch.tensor(all_labels, dtype=torch.long),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_head(
    data_root: Optional[Path],
    output_dir: Path,
    gigapath_model=None,
    hf_token: Optional[str] = None,
    epochs: int         = 10,
    batch_size: int     = 128,
    lr: float           = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float     = 0.15,
    test_frac: float    = 0.10,
    max_samples: Optional[int] = None,
    seed: int           = 42,
) -> dict:
    """
    Train the GigaPath classification head.

    If data_root is None or no images are found, trains on synthetic random
    embeddings for smoke-testing purposes.

    Returns:
        Training report dict (also saved to output_dir/training_report.json).
    """
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix

    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"GigaPathHead training on device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GigaPath ────────────────────────────────────────────────────
    if gigapath_model is None:
        try:
            from ml.models.gigapath_loader import load_gigapath, build_transform
            token = hf_token or os.getenv("HF_TOKEN", "")
            gigapath_model = load_gigapath(hf_token=token or None).to(device)
            gigapath_model.eval()
            transform = build_transform(augment=False)
            log.info("GigaPath loaded from HuggingFace")
        except Exception as e:
            log.warning(f"Could not load GigaPath ({e}) — using random embeddings for smoke test")
            gigapath_model = None
            transform      = None

    # ── Extract / generate embeddings ────────────────────────────────────
    if gigapath_model is not None and data_root is not None and data_root.exists():
        log.info(f"Extracting embeddings from {data_root} ...")
        t_extract = time.perf_counter()
        embeddings, labels = extract_embeddings_from_folder(
            data_root, gigapath_model, transform,
            batch_size=min(64, batch_size),
            max_samples=max_samples,
            device=device,
        )
        log.info(f"Extracted {len(embeddings)} embeddings in {time.perf_counter() - t_extract:.1f}s")
    else:
        log.warning("Generating random embeddings for smoke-test training")
        n_per_class = max(10, (max_samples or 500) // N_CLASSES)
        embeddings = torch.randn(N_CLASSES * n_per_class, EMBEDDING_DIM)
        embeddings = F.normalize(embeddings, dim=-1)
        labels     = torch.tensor([i for i in range(N_CLASSES) for _ in range(n_per_class)], dtype=torch.long)

    N = len(embeddings)
    indices = list(range(N))
    random.shuffle(indices)

    n_test = max(1, int(N * test_frac))
    n_val  = max(1, int(N * val_frac))
    test_idx = indices[:n_test]
    val_idx  = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_ds = EmbeddingDataset(embeddings[train_idx], labels[train_idx])
    val_ds   = EmbeddingDataset(embeddings[val_idx],   labels[val_idx])
    test_ds  = EmbeddingDataset(embeddings[test_idx],  labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    log.info(f"Dataset: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # ── Build model ──────────────────────────────────────────────────────
    model     = GigaPathHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"GigaPathHead trainable params: {trainable_params:,}")

    # ── Training ──────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    t_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for emb_batch, lbl_batch in train_loader:
            emb_batch = emb_batch.to(device)
            lbl_batch = lbl_batch.to(device)
            optimiser.zero_grad()
            logits = model(emb_batch)
            loss   = criterion(logits, lbl_batch)
            loss.backward()
            optimiser.step()
            scheduler.step()
            train_loss += loss.item() * emb_batch.shape[0]

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for emb_batch, lbl_batch in val_loader:
                emb_batch = emb_batch.to(device)
                lbl_batch = lbl_batch.to(device)
                logits    = model(emb_batch)
                val_loss  += criterion(logits, lbl_batch).item() * emb_batch.shape[0]
                val_correct += (logits.argmax(dim=-1) == lbl_batch).sum().item()

        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        log.info(f"Epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "giga_head.pt")
            log.info(f"  Saved best model (val_acc={val_acc:.4f})")

    elapsed = round(time.perf_counter() - t_train, 1)

    # ── Test set evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "giga_head.pt", map_location=device))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for emb_batch, lbl_batch in test_loader:
            preds = model(emb_batch.to(device)).argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(lbl_batch.tolist())

    test_acc  = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    per_class_f1 = f1_score(all_true, all_preds, average=None, labels=list(range(N_CLASSES))).tolist()
    macro_f1     = f1_score(all_true, all_preds, average="macro")
    conf_matrix  = confusion_matrix(all_true, all_preds, labels=list(range(N_CLASSES))).tolist()

    log.info(f"Test accuracy : {test_acc:.4f} ({test_acc:.1%})")
    log.info(f"Macro F1      : {macro_f1:.4f}")
    for i, (cls, f1) in enumerate(zip(TISSUE_CLASSES, per_class_f1)):
        log.info(f"  {cls}: F1={f1:.3f}")

    # ── Save report ───────────────────────────────────────────────────────
    report = {
        "model":        "GigaPathHead",
        "architecture": "Linear(1536→512)+BN+ReLU+Dropout(0.2)→Linear(512→5)",
        "backbone":     "prov-gigapath/prov-gigapath (frozen)",
        "trainable_params": trainable_params,
        "total_samples": N,
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "test_samples":  len(test_ds),
        "hyperparams": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay},
        "results": {
            "test_accuracy": round(test_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": {cls: round(f, 4) for cls, f in zip(TISSUE_CLASSES, per_class_f1)},
            "confusion_matrix": conf_matrix,
            "best_val_accuracy": round(best_val_acc, 4),
        },
        "history":   history,
        "elapsed_s": elapsed,
        "hardware":  f"AMD MI300X · ROCm" if device.type == "cuda" else "CPU",
        "classes":   TISSUE_CLASSES,
    }

    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Training report saved to {output_dir}/training_report.json")

    # ── Save confusion matrix as PNG ──────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array(conf_matrix)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels([c.replace("_", "\n") for c in TISSUE_CLASSES], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c.replace("_", " ") for c in TISSUE_CLASSES], fontsize=8)
        ax.set_title(f"GigaPathHead — Test Accuracy {test_acc:.1%} | Macro F1 {macro_f1:.3f}", fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        log.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    except Exception as e:
        log.warning(f"Could not save confusion matrix PNG: {e}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GigaPath MLP classification head on LC25000.")
    p.add_argument("--data_root",   default=None,   help="Path to LC25000 root directory")
    p.add_argument("--output_dir",  default="aob/ml/models/checkpoints/giga_head")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Max total samples to use (e.g. 5000 for quick run). None = all.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    report = train_head(
        data_root=Path(args.data_root) if args.data_root else None,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
    print(f"\nTest accuracy: {report['results']['test_accuracy']:.1%}")
    print(f"Macro F1:      {report['results']['macro_f1']:.3f}")
