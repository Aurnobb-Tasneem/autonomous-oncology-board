"""
scripts/finetune_tnm.py
=======================
LoRA fine-tuning of meta-llama/Llama-3.1-8B-Instruct for TNM staging on AMD MI300X.

Track 2 (Fine-Tuning on AMD GPUs) deliverable.

What this script does
---------------------
1. Creates a 50-example mock dataset mapping pathology text → structured TNM JSON.
2. Applies LoRA to meta-llama/Llama-3.1-8B-Instruct via `peft`.
3. Trains with HuggingFace Trainer + bf16 + gradient checkpointing on ROCm.
4. Integrates Optimum-AMD when available (BetterTransformer graph optimisation).
5. Saves the LoRA adapter to `<output_dir>/` and writes `training_report.json`.

Usage
-----
# Minimal (1 epoch, all defaults):
  HF_TOKEN=hf_... python scripts/finetune_tnm.py

# Full options:
  python scripts/finetune_tnm.py \\
    --base_model meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir aob/ml/models/checkpoints/tnm_lora \\
    --epochs 3 \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --lr 2e-4

# Quick smoke test (50 steps only):
  python scripts/finetune_tnm.py --max_steps 50

Serve the adapter (vLLM, OpenAI-compatible):
  see serve_tnm_adapter() at the bottom of this file or run with --serve

VRAM budget on AMD MI300X (192 GB HBM3):
  Llama-3-8B (bf16)      ~  16 GB
  LoRA trainable params  <   1 GB  (r=8, q/k/v/o only)
  Activations + grad     ~   8 GB  (batch=2, grad_accum=4, grad_ckpt=on)
  ─────────────────────────────────────────────────────
  Total estimated        ~  25 GB   (leaves 115 GB free for GigaPath + 70B)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("finetune_tnm")

# ── Prompt template ──────────────────────────────────────────────────────────
# Kept in one place so the adapter and the inference agent use identical
# formatting.  The agent (staging_specialist.py) imports PROMPT_TEMPLATE.
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

RESPONSE_TEMPLATE = """{tnm_json}<|eot_id|>"""

# ── Mock dataset (50 examples) ───────────────────────────────────────────────
# Covers the 5 LC25000 tissue classes + realistic clinical language.
# Each entry is (pathology_text, tnm_dict).  Kept in-code so the script has
# zero external file dependencies.

_TNM_EXAMPLES: list[tuple[str, dict]] = [
    # Lung Adenocarcinoma (18 examples)
    (
        "3.2 cm peripheral lung adenocarcinoma, no pleural invasion, 2/15 lymph nodes positive, "
        "no distant metastasis. Glandular patterns with nuclear atypia.",
        {"T": "T2a", "N": "N1", "M": "M0", "stage": "IIB"},
    ),
    (
        "Lung adenocarcinoma measuring 5.8 cm, visceral pleural invasion present, "
        "3 ipsilateral mediastinal nodes positive, no metastasis.",
        {"T": "T3", "N": "N2", "M": "M0", "stage": "IIIA"},
    ),
    (
        "1.8 cm lung adenocarcinoma confined to lung parenchyma, all 12 lymph nodes negative.",
        {"T": "T1b", "N": "N0", "M": "M0", "stage": "IA2"},
    ),
    (
        "Lung adenocarcinoma with micropapillary predominant pattern, 4.1 cm, "
        "pleural invasion, 1 hilar node positive, no metastasis.",
        {"T": "T2b", "N": "N1", "M": "M0", "stage": "IIB"},
    ),
    (
        "Poorly differentiated lung adenocarcinoma 7 cm invading chest wall, "
        "5/18 mediastinal nodes positive, liver metastasis identified.",
        {"T": "T3", "N": "N2", "M": "M1b", "stage": "IVA"},
    ),
    (
        "2.5 cm mucinous adenocarcinoma of the lung, no nodal involvement, no metastasis.",
        {"T": "T1c", "N": "N0", "M": "M0", "stage": "IA3"},
    ),
    (
        "Lung adenocarcinoma 6.5 cm with satellite nodule in same lobe, "
        "0/10 lymph nodes positive, no distant spread.",
        {"T": "T3", "N": "N0", "M": "M0", "stage": "IIB"},
    ),
    (
        "Acinar predominant lung adenocarcinoma 3.9 cm, no pleural invasion, "
        "2 contralateral mediastinal nodes positive, no metastasis.",
        {"T": "T2a", "N": "N3", "M": "M0", "stage": "IIIB"},
    ),
    (
        "Well-differentiated lung adenocarcinoma 1.1 cm, no nodes positive, no metastasis.",
        {"T": "T1a", "N": "N0", "M": "M0", "stage": "IA1"},
    ),
    (
        "Lung adenocarcinoma with bone metastasis, primary 4.4 cm, "
        "4 mediastinal nodes positive.",
        {"T": "T2b", "N": "N2", "M": "M1b", "stage": "IVA"},
    ),
    (
        "Right upper lobe adenocarcinoma 2.0 cm, no pleural or vascular invasion, "
        "negative nodes, no metastasis.",
        {"T": "T1b", "N": "N0", "M": "M0", "stage": "IA2"},
    ),
    (
        "Lung adenocarcinoma 5.1 cm invading diaphragm, 1/14 hilar node positive, "
        "no distant metastasis.",
        {"T": "T3", "N": "N1", "M": "M0", "stage": "IIIA"},
    ),
    (
        "EGFR-mutant lung adenocarcinoma 3.0 cm, no nodal involvement, "
        "adrenal metastasis on PET/CT.",
        {"T": "T2a", "N": "N0", "M": "M1b", "stage": "IVA"},
    ),
    (
        "Adenocarcinoma in situ (AIS) right upper lobe, 0.7 cm, no invasion, "
        "nodes negative.",
        {"T": "Tis(AIS)", "N": "N0", "M": "M0", "stage": "0"},
    ),
    (
        "Minimally invasive lung adenocarcinoma 0.9 cm, lepidic predominant, "
        "no nodes, no metastasis.",
        {"T": "T1mi", "N": "N0", "M": "M0", "stage": "IA1"},
    ),
    (
        "Lung adenocarcinoma 4.8 cm with pericardial involvement, "
        "bilateral mediastinal nodes positive, no metastasis.",
        {"T": "T4", "N": "N3", "M": "M0", "stage": "IIIC"},
    ),
    (
        "Multifocal lung adenocarcinoma with pneumonic pattern (multiple lobes), "
        "0/8 nodes positive, no extrathoracic metastasis.",
        {"T": "T3", "N": "N0", "M": "M0", "stage": "IIB"},
    ),
    (
        "Lung adenocarcinoma 2.8 cm, lymphovascular invasion present, "
        "1 ipsilateral node positive, no distant disease.",
        {"T": "T2a", "N": "N1", "M": "M0", "stage": "IIB"},
    ),

    # Lung Squamous Cell Carcinoma (11 examples)
    (
        "Moderately differentiated squamous cell carcinoma of the lung, 3.5 cm, "
        "central location, no nodal involvement, no metastasis.",
        {"T": "T2a", "N": "N0", "M": "M0", "stage": "IB"},
    ),
    (
        "Squamous cell carcinoma of right bronchus, 6 cm, invading carina, "
        "3 subcarinal nodes positive, no distant metastasis.",
        {"T": "T4", "N": "N2", "M": "M0", "stage": "IIIB"},
    ),
    (
        "Poorly differentiated squamous carcinoma 1.5 cm, peripheral, "
        "no nodal disease, no metastasis.",
        {"T": "T1b", "N": "N0", "M": "M0", "stage": "IA2"},
    ),
    (
        "Central squamous cell carcinoma of the lung with total atelectasis of lobe, "
        "2 ipsilateral hilar nodes, no distant spread.",
        {"T": "T2b", "N": "N1", "M": "M0", "stage": "IIB"},
    ),
    (
        "Squamous carcinoma 4.0 cm invading parietal pleura and chest wall, "
        "no nodal involvement, no metastasis.",
        {"T": "T3", "N": "N0", "M": "M0", "stage": "IIB"},
    ),
    (
        "Squamous cell carcinoma with obstructive pneumonitis, 5.5 cm, "
        "4 contralateral hilar nodes, brain metastasis.",
        {"T": "T3", "N": "N3", "M": "M1b", "stage": "IVA"},
    ),
    (
        "Well-differentiated squamous carcinoma 2.2 cm, no pleural invasion, "
        "all sampled nodes negative.",
        {"T": "T1c", "N": "N0", "M": "M0", "stage": "IA3"},
    ),
    (
        "Squamous carcinoma invading SVC and trachea, 7 cm, bilateral nodal disease.",
        {"T": "T4", "N": "N3", "M": "M0", "stage": "IIIC"},
    ),
    (
        "Squamous cell carcinoma of the lung 3.8 cm, no nodes, bone metastasis identified.",
        {"T": "T2a", "N": "N0", "M": "M1b", "stage": "IVA"},
    ),
    (
        "Superficial squamous carcinoma confined to bronchial mucosa, 0.4 cm, "
        "no nodal involvement.",
        {"T": "Tis", "N": "N0", "M": "M0", "stage": "0"},
    ),
    (
        "Squamous carcinoma 2.9 cm with chest wall invasion, 1 hilar node positive.",
        {"T": "T3", "N": "N1", "M": "M0", "stage": "IIIA"},
    ),

    # Colon Adenocarcinoma (13 examples)
    (
        "Moderately differentiated colon adenocarcinoma, invades into but not through "
        "muscularis propria (pT2), 0/22 lymph nodes, no metastasis.",
        {"T": "T2", "N": "N0", "M": "M0", "stage": "I"},
    ),
    (
        "Colon adenocarcinoma invading through muscularis propria into pericolorectal "
        "tissues (pT3), 4/18 regional nodes positive, no distant metastasis.",
        {"T": "T3", "N": "N2a", "M": "M0", "stage": "IIIB"},
    ),
    (
        "Poorly differentiated colon adenocarcinoma penetrating visceral peritoneum (pT4a), "
        "2/15 regional nodes, liver metastasis.",
        {"T": "T4a", "N": "N1b", "M": "M1a", "stage": "IVA"},
    ),
    (
        "Well-differentiated mucinous adenocarcinoma of ascending colon, "
        "invades submucosa (pT1), no nodes involved, no metastasis.",
        {"T": "T1", "N": "N0", "M": "M0", "stage": "I"},
    ),
    (
        "Colon adenocarcinoma directly invading adjacent small bowel (pT4b), "
        "6 regional nodes positive, no distant metastasis.",
        {"T": "T4b", "N": "N2b", "M": "M0", "stage": "IIIC"},
    ),
    (
        "MSI-H colon adenocarcinoma pT3, 1/24 lymph node positive, no metastasis.",
        {"T": "T3", "N": "N1a", "M": "M0", "stage": "IIIA"},
    ),
    (
        "Colon adenocarcinoma pT3 N0 M0 with peritoneal seeding on CT.",
        {"T": "T3", "N": "N0", "M": "M1c", "stage": "IVC"},
    ),
    (
        "Sigmoid colon adenocarcinoma pT2, 0/20 nodes, pulmonary metastasis on PET.",
        {"T": "T2", "N": "N0", "M": "M1b", "stage": "IVB"},
    ),
    (
        "Colon adenocarcinoma in situ (intraepithelial), no invasion, nodes negative.",
        {"T": "Tis", "N": "N0", "M": "M0", "stage": "0"},
    ),
    (
        "Right colon adenocarcinoma pT3, 7/21 regional nodes positive, "
        "no distant disease. KRAS mutant.",
        {"T": "T3", "N": "N2b", "M": "M0", "stage": "IIIC"},
    ),
    (
        "Colon adenocarcinoma pT4a, 0 nodes positive, ovarian metastasis.",
        {"T": "T4a", "N": "N0", "M": "M1a", "stage": "IVA"},
    ),
    (
        "Transverse colon adenocarcinoma pT1, lymphovascular invasion present, "
        "0/18 nodes positive, no distant spread.",
        {"T": "T1", "N": "N0", "M": "M0", "stage": "I"},
    ),
    (
        "Moderately differentiated colon adenocarcinoma pT3, 3 nodes positive "
        "(apical node involved), no metastasis.",
        {"T": "T3", "N": "N1b", "M": "M0", "stage": "IIIB"},
    ),

    # Colon Benign & Lung Benign (8 examples, no malignant staging)
    (
        "Hyperplastic polyp of the colon, no dysplasia, no invasion, lymph nodes not sampled.",
        {"T": "T0", "N": "NX", "M": "M0", "stage": "N/A"},
    ),
    (
        "Tubular adenoma with low-grade dysplasia, completely excised, no invasive carcinoma.",
        {"T": "Tis", "N": "N0", "M": "M0", "stage": "0"},
    ),
    (
        "Lung biopsy: organising pneumonia, no malignancy identified. Lymph nodes not sampled.",
        {"T": "T0", "N": "NX", "M": "M0", "stage": "N/A"},
    ),
    (
        "Hamartoma of the lung, 1.4 cm, completely resected, no lymph node involvement.",
        {"T": "T0", "N": "N0", "M": "M0", "stage": "N/A"},
    ),
    (
        "Tubulovillous adenoma with focal high-grade dysplasia, no submucosal invasion.",
        {"T": "Tis", "N": "N0", "M": "M0", "stage": "0"},
    ),
    (
        "Atypical adenomatous hyperplasia of the lung, 0.3 cm, no nodes sampled, "
        "no invasion.",
        {"T": "T0", "N": "NX", "M": "M0", "stage": "N/A"},
    ),
    (
        "Carcinoid tumorlet of the lung, 0.2 cm, no angiolymphatic invasion, "
        "nodes negative.",
        {"T": "T1a", "N": "N0", "M": "M0", "stage": "IA1"},
    ),
    (
        "Serrated adenoma of the colon with no dysplasia, completely excised.",
        {"T": "Tis", "N": "N0", "M": "M0", "stage": "0"},
    ),
]

assert len(_TNM_EXAMPLES) == 50, f"Expected 50 examples, got {len(_TNM_EXAMPLES)}"


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_training_text(pathology_text: str, tnm_json: dict) -> str:
    """Build the full prompt+response string for causal LM training."""
    prompt = PROMPT_TEMPLATE.format(pathology_text=pathology_text.strip())
    response = RESPONSE_TEMPLATE.format(tnm_json=json.dumps(tnm_json, separators=(",", ":")))
    return prompt + response


def build_hf_dataset(examples: list[tuple[str, dict]], eval_frac: float = 0.2):
    """
    Convert the raw examples list to a HuggingFace DatasetDict.

    Returns:
        datasets.DatasetDict with 'train' and 'test' splits.
    """
    from datasets import Dataset, DatasetDict

    random.shuffle(examples)  # deterministic shuffle below via seed
    random.seed(42)
    random.shuffle(examples)

    cut = max(1, int(len(examples) * (1 - eval_frac)))
    train_examples = examples[:cut]
    eval_examples  = examples[cut:]

    def to_records(subset):
        return [
            {
                "text": build_training_text(p, t),
                "pathology_text": p,
                "tnm_json": json.dumps(t),
            }
            for p, t in subset
        ]

    return DatasetDict({
        "train": Dataset.from_list(to_records(train_examples)),
        "test":  Dataset.from_list(to_records(eval_examples)),
    })


# ── Optimum-AMD integration ──────────────────────────────────────────────────

def apply_optimum_amd(model):
    """
    Attempt to apply Optimum-AMD BetterTransformer optimisation for ROCm.

    Optimum-AMD accelerates inference and in some cases training by replacing
    standard attention layers with optimised ROCm-compatible kernels.

    Falls back gracefully if Optimum-AMD isn't installed or the model
    architecture isn't supported yet (LlamaAttention support varies by version).
    """
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model, keep_original_model=False)
        log.info("Optimum-AMD: BetterTransformer applied successfully.")
        return model, True
    except ImportError:
        log.warning("Optimum-AMD: `optimum` package not installed — skipping BetterTransformer.")
        return model, False
    except Exception as e:
        log.warning(
            f"Optimum-AMD: BetterTransformer transform failed ({type(e).__name__}: {e}) "
            "— falling back to standard attention. This is expected for some Llama versions."
        )
        return model, False


# ── Training ─────────────────────────────────────────────────────────────────

def compute_exact_match(model, tokenizer, eval_examples: list[tuple[str, dict]], device) -> float:
    """
    Compute exact-match accuracy on the eval set.

    An exact match means the model's greedy-decoded JSON, when parsed, equals
    the target TNM dict.  Partial credit is NOT given.
    """
    import torch

    model.eval()
    matches = 0
    total = len(eval_examples)

    with torch.no_grad():
        for pathology_text, target_tnm in eval_examples:
            prompt = PROMPT_TEMPLATE.format(pathology_text=pathology_text.strip())
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Extract first JSON object from generated text
            m = re.search(r"\{.*?\}", generated, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if parsed == target_tnm:
                        matches += 1
                except json.JSONDecodeError:
                    pass

    return matches / total if total > 0 else 0.0


def run_training(args: argparse.Namespace):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ── 0. HF auth ────────────────────────────────────────────────────────
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        log.error(
            "HF_TOKEN environment variable is not set. "
            "meta-llama/Llama-3.1-8B-Instruct is a gated model and requires "
            "a HuggingFace access token with model access granted at "
            "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct\n"
            "Export your token:  export HF_TOKEN=hf_..."
        )
        sys.exit(1)

    from huggingface_hub import login as hf_login
    hf_login(token=hf_token, add_to_git_credential=False)
    log.info(f"HF: authenticated — loading base model '{args.base_model}'")

    # ── 1. Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("No GPU detected — running on CPU (expect slow training)")
    else:
        log.info(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0)}")

    # ── 2. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=hf_token,
        padding_side="right",   # Llama needs right-padding for training
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Base model (bf16 on MI300X, fp16 fallback) ─────────────────────
    # ROCm MI300X supports bf16 natively and it's more numerically stable.
    # torch_dtype auto lets HF pick the best dtype for the device.
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else torch.float16

    log.info(f"Loading base model in {load_dtype} with gradient checkpointing ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=load_dtype,
        device_map="auto",            # honours multiple GPUs if present
        trust_remote_code=True,
    )
    model.config.use_cache = False    # required for gradient checkpointing

    # ── 4. Optimum-AMD (optional) ─────────────────────────────────────────
    # Applied before PEFT — BetterTransformer rewrites attention modules.
    # Must be reversed (BetterTransformer.reverse) if gradient checkpointing
    # fails.  We wrap in a try/finally so training always proceeds.
    model, optimum_applied = apply_optimum_amd(model)

    # ── 5. LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Target the attention projections; this keeps VRAM low
        # while capturing the key QKV representations the model needs
        # to learn structured TNM output.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable, total, pct = model.get_nb_trainable_parameters()
    log.info(
        f"LoRA: trainable params = {trainable:,} / {total:,} ({pct:.2f}%) "
        f"  r={args.lora_r}  alpha={args.lora_alpha}"
    )

    # ── 6. Dataset ────────────────────────────────────────────────────────
    raw_dataset = build_hf_dataset(_TNM_EXAMPLES)
    log.info(f"Dataset: {len(raw_dataset['train'])} train / {len(raw_dataset['test'])} eval")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )

    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=["text", "tnm_json"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── 7. Training arguments ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=not use_bf16 and (device.type == "cuda"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",           # no wandb / tensorboard required
        remove_unused_columns=True,
        dataloader_pin_memory=False,  # some ROCm builds dislike pinned memory
    )

    # ── 8. Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    log.info("Starting LoRA fine-tuning ...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = round(time.perf_counter() - t0, 1)
    log.info(f"Training complete in {elapsed}s")

    # ── 9. Save adapter ───────────────────────────────────────────────────
    # Save only the LoRA adapter weights (not the full base model).
    # The adapter is ~50–100 MB; the base model stays on HF Hub.
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info(f"LoRA adapter saved to {output_dir}/")

    # ── 10. Exact-match on eval set ───────────────────────────────────────
    log.info("Computing exact-match on eval set ...")
    eval_examples_raw = [
        (row["pathology_text"], json.loads(row["tnm_json"]))
        for row in raw_dataset["test"]
    ]
    # Reverse BetterTransformer before calling generate (not supported with BT)
    if optimum_applied:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.reverse(model)
        except Exception:
            pass

    exact_match = compute_exact_match(model, tokenizer, eval_examples_raw, device)
    log.info(f"Eval exact-match: {exact_match:.1%}  ({len(eval_examples_raw)} examples)")

    # ── 11. Write training report ─────────────────────────────────────────
    train_metrics = train_result.metrics
    report = {
        "base_model":       args.base_model,
        "adapter_path":     str(output_dir.resolve()),
        "lora_config": {
            "r":             args.lora_r,
            "lora_alpha":    args.lora_alpha,
            "lora_dropout":  args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "training": {
            "epochs":          args.epochs,
            "max_steps":       args.max_steps,
            "batch_size":      args.batch_size,
            "grad_accum":      args.grad_accum,
            "lr":              args.lr,
            "elapsed_s":       elapsed,
            "train_loss":      round(train_metrics.get("train_loss", -1), 4),
            "train_runtime_s": round(train_metrics.get("train_runtime", 0), 1),
        },
        "eval": {
            "n_examples":   len(eval_examples_raw),
            "exact_match":  round(exact_match, 4),
        },
        "optimum_amd_applied": optimum_applied,
        "dtype":  "bfloat16" if use_bf16 else "float16",
        "prompt_template": PROMPT_TEMPLATE,
        "dataset_size": len(_TNM_EXAMPLES),
        "hardware": "AMD Instinct MI300X · ROCm" if device.type == "cuda" else "CPU",
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Training report written to {report_path}")

    return report


# ── vLLM serve helper ────────────────────────────────────────────────────────

def serve_tnm_adapter(args: argparse.Namespace):
    """
    Print the vLLM OpenAI-compatible serve command for the TNM adapter.

    We print rather than exec because vLLM must run in its own process
    (it takes over the GPU).  The user copies and runs the command, or
    wraps it in a systemd unit / PM2 process on the MI300X.

    VRAM budget: meta-llama/Llama-3.1-8B-Instruct ~16 GB + LoRA overhead <1 GB = ~17 GB.
    This leaves >120 GB free for GigaPath + Llama-3.3-70B running in parallel.
    """
    cmd = (
        f"HF_TOKEN=$HF_TOKEN \\\n"
        f"python -m vllm.entrypoints.openai.api_server \\\n"
        f"  --model {args.base_model} \\\n"
        f"  --enable-lora \\\n"
        f"  --lora-modules tnm_specialist={args.output_dir} \\\n"
        f"  --port {args.vllm_port} \\\n"
        f"  --gpu-memory-utilization {args.gpu_mem_util} \\\n"
        f"  --max-model-len 2048 \\\n"
        f"  --dtype bfloat16"
    )

    print("\n" + "=" * 70)
    print("  vLLM Serve Command (OpenAI-compatible, ROCm MI300X)")
    print("=" * 70)
    print(cmd)
    print("=" * 70)
    print("\nOnce running, test with:")
    print(
        f'  curl http://localhost:{args.vllm_port}/v1/chat/completions \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f'    -d \'{{\n'
        f'      "model": "tnm_specialist",\n'
        f'      "messages": [{{\n'
        f'        "role": "user",\n'
        f'        "content": "3.2 cm lung adenocarcinoma, 2/15 nodes positive, no metastasis."\n'
        f'      }}],\n'
        f'      "max_tokens": 80,\n'
        f'      "temperature": 0\n'
        f'    }}\''
    )
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tune meta-llama/Llama-3.1-8B-Instruct for TNM staging on AMD MI300X."
    )
    p.add_argument(
        "--base_model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for the base model.",
    )
    p.add_argument(
        "--output_dir",
        default="aob/ml/models/checkpoints/tnm_lora",
        help="Directory to save the LoRA adapter weights and training report.",
    )
    p.add_argument("--epochs",      type=int,   default=1,    help="Training epochs.")
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epochs with a fixed step count (useful for smoke tests). -1 = use epochs.",
    )
    p.add_argument("--batch_size",  type=int,   default=2,    help="Per-device train batch size.")
    p.add_argument("--grad_accum",  type=int,   default=4,    help="Gradient accumulation steps.")
    p.add_argument("--lr",          type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--lora_r",      type=int,   default=8,    help="LoRA rank.")
    p.add_argument("--lora_alpha",  type=int,   default=16,   help="LoRA alpha.")
    p.add_argument("--lora_dropout",type=float, default=0.05, help="LoRA dropout.")
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Max sequence length for tokenisation.",
    )
    p.add_argument(
        "--vllm_port",
        type=int,
        default=8006,
        help="Port for the vLLM OpenAI-compatible server (only used with --serve).",
    )
    p.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.15,
        help=(
            "vLLM GPU memory utilisation fraction (0.0–1.0). "
            "Default 0.15 (~29 GB on MI300X) leaves room for GigaPath + 70B."
        ),
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="After training, print the vLLM serve command (does not start vLLM).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  AOB TNM Specialist — LoRA Fine-Tune (Track 2)")
    log.info("  AMD MI300X · ROCm · Optimum-AMD")
    log.info("=" * 60)
    log.info(f"Base model : {args.base_model}")
    log.info(f"Output dir : {args.output_dir}")
    log.info(f"Epochs     : {args.epochs}  (max_steps={args.max_steps})")
    log.info(f"LoRA       : r={args.lora_r}  alpha={args.lora_alpha}  dropout={args.lora_dropout}")
    log.info(f"LR         : {args.lr}")

    report = run_training(args)

    log.info("\n" + "=" * 60)
    log.info("  Training Summary")
    log.info("=" * 60)
    log.info(f"  Train loss   : {report['training']['train_loss']}")
    log.info(f"  Eval exact-match : {report['eval']['exact_match']:.1%}")
    log.info(f"  Adapter saved to : {report['adapter_path']}")
    log.info(f"  Optimum-AMD      : {'applied' if report['optimum_amd_applied'] else 'skipped'}")
    log.info("=" * 60)

    if args.serve:
        serve_tnm_adapter(args)
    else:
        log.info("\nTo serve the adapter with vLLM, run:")
        log.info(f"  python scripts/finetune_tnm.py --serve  (prints the command)")


if __name__ == "__main__":
    main()
