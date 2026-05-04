"""
ml/training/lora_trainer.py
============================
Generic LoRA adapter trainer for any (text → JSON) task on AMD MI300X.

Design
------
The training logic extracted from scripts/finetune_tnm.py into a reusable
function so that biomarker, treatment, and any future specialist adapters
can be trained with minimal boilerplate.

Usage
-----
    from ml.training.lora_trainer import LoRATrainingSpec, train_lora_adapter

    spec = LoRATrainingSpec(
        task_name="biomarker",
        prompt_template=PROMPT_TEMPLATE,
        examples=EXAMPLES,
        output_schema_keys={"tests_required", "gated_therapies", "rationale"},
        output_dir=Path("aob/ml/models/checkpoints/biomarker_lora"),
    )
    report = train_lora_adapter(spec)

VRAM budget on AMD MI300X (192 GB HBM3) when sharing with other services:
    Llama-3.1-8B (bf16)   ~  16 GB
    LoRA trainable params  <   1 GB  (r=8, q/k/v/o only)
    Activations + grad     ~   8 GB  (batch=2, grad_accum=4, grad_ckpt=on)
    Total per adapter      ~  25 GB  — leaves 115+ GB free for GigaPath + 70B
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Prompt / response wrappers ────────────────────────────────────────────────
SYSTEM_HEADER = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a board-certified oncological specialist.
Given the input text, output ONLY a JSON object with the keys specified.
Do not output any explanation. Output only valid JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

SYSTEM_FOOTER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
RESPONSE_EOS  = "<|eot_id|>"


def build_default_prompt_template(task_description: str, schema_description: str) -> str:
    """Build a Llama-3 prompt template for any JSON-output clinical task."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a board-certified oncological specialist.\n"
        f"{task_description}\n"
        f"Output ONLY a JSON object with these keys: {schema_description}.\n"
        f"Do not output any explanation. Output only valid JSON.\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{{input_text}}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LoRATrainingSpec:
    """
    Complete specification for one LoRA adapter training run.

    Attributes:
        task_name:           Short identifier: "tnm" | "biomarker" | "treatment"
        prompt_template:     Llama-3 chat template with {input_text} placeholder.
        examples:            List of (input_text, target_json_dict) pairs.
        output_schema_keys:  Required JSON keys — used for exact-match eval.
        output_dir:          Directory for adapter weights + training_report.json.
    """
    task_name:           str
    prompt_template:     str
    examples:            list     # list[tuple[str, dict]]
    output_schema_keys:  set
    output_dir:          Path

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_schema_keys = set(self.output_schema_keys)


@dataclass
class TrainingReport:
    """Serialisable record of one LoRA training run."""
    task_name:          str
    base_model:         str
    adapter_path:       str
    dataset_size:       int
    eval_size:          int
    lora_config:        dict
    training:           dict
    eval:               dict
    optimum_applied:    bool
    dtype:              str
    hardware:           str
    prompt_template:    str

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Training report saved to {path}")


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_training_text(
    input_text: str,
    target_json: dict,
    prompt_template: str,
) -> str:
    """Construct full prompt+response string for causal LM training."""
    prompt   = prompt_template.format(input_text=input_text.strip())
    response = json.dumps(target_json, separators=(",", ":")) + RESPONSE_EOS
    return prompt + response


def build_hf_dataset(
    examples: list,           # list[tuple[str, dict]]
    prompt_template: str,
    eval_frac: float = 0.2,
    seed: int = 42,
):
    """
    Convert raw examples to a HuggingFace DatasetDict (train / test).

    Returns:
        datasets.DatasetDict with 'train' and 'test' splits.
    """
    from datasets import Dataset, DatasetDict

    shuffled = list(examples)
    random.seed(seed)
    random.shuffle(shuffled)

    cut = max(1, int(len(shuffled) * (1 - eval_frac)))
    train_ex = shuffled[:cut]
    eval_ex  = shuffled[cut:]

    def to_records(subset):
        return [
            {
                "text":       build_training_text(inp, tgt, prompt_template),
                "input_text": inp,
                "target_json": json.dumps(tgt),
            }
            for inp, tgt in subset
        ]

    return DatasetDict({
        "train": Dataset.from_list(to_records(train_ex)),
        "test":  Dataset.from_list(to_records(eval_ex)),
    })


# ── Optimum-AMD ───────────────────────────────────────────────────────────────

def _apply_optimum_amd(model):
    """
    Attempt to apply BetterTransformer via Optimum-AMD. Falls back silently.
    Returns (model, applied: bool).
    """
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model, keep_original_model=False)
        log.info("Optimum-AMD: BetterTransformer applied.")
        return model, True
    except ImportError:
        log.warning("Optimum-AMD: `optimum` not installed — skipping.")
        return model, False
    except Exception as e:
        log.warning(f"Optimum-AMD: BetterTransformer failed ({e}) — skipping.")
        return model, False


def _reverse_optimum_amd(model, applied: bool):
    """Reverse BetterTransformer before calling model.generate()."""
    if not applied:
        return model
    try:
        from optimum.bettertransformer import BetterTransformer
        return BetterTransformer.reverse(model)
    except Exception:
        return model


# ── Exact-match evaluation ────────────────────────────────────────────────────

def compute_exact_match(
    model,
    tokenizer,
    eval_examples: list,  # list[tuple[str, dict]]
    prompt_template: str,
    output_schema_keys: set,
    device,
    max_new_tokens: int = 128,
) -> dict:
    """
    Compute exact-match accuracy on the eval set.

    Returns:
        dict with keys: exact_match (float), schema_compliance (float),
        n_examples (int), per_key_accuracy (dict).
    """
    import torch

    model.eval()
    exact_matches   = 0
    schema_ok       = 0
    key_hits: dict  = {k: 0 for k in output_schema_keys}
    total           = len(eval_examples)

    with torch.no_grad():
        for input_text, target_json in eval_examples:
            prompt = prompt_template.format(input_text=input_text.strip())
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            m = re.search(r"\{.*?\}", generated, re.DOTALL)
            parsed = None
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

            if parsed is not None:
                schema_ok += 1
                if parsed == target_json:
                    exact_matches += 1
                for key in output_schema_keys:
                    if key in parsed:
                        key_hits[key] += 1

    per_key = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in key_hits.items()}
    return {
        "exact_match":      round(exact_matches / total, 4) if total > 0 else 0.0,
        "schema_compliance": round(schema_ok / total, 4) if total > 0 else 0.0,
        "n_examples":       total,
        "per_key_accuracy": per_key,
    }


# ── Main trainer ──────────────────────────────────────────────────────────────

def train_lora_adapter(
    spec: LoRATrainingSpec,
    base_model: str    = "meta-llama/Llama-3.1-8B-Instruct",
    lora_r: int        = 8,
    lora_alpha: int    = 16,
    lora_dropout: float = 0.05,
    epochs: int        = 1,
    max_steps: int     = -1,
    batch_size: int    = 2,
    grad_accum: int    = 4,
    lr: float          = 2e-4,
    max_seq_len: int   = 512,
    eval_frac: float   = 0.2,
    seed: int          = 42,
    hf_token: Optional[str] = None,
) -> TrainingReport:
    """
    Train a LoRA adapter on a clinical JSON extraction task.

    This is the single source of truth for LoRA training across all AOB
    specialist adapters (TNM, biomarker, treatment, and future tasks).

    Args:
        spec:        Task specification — see LoRATrainingSpec.
        base_model:  HuggingFace model ID for the base model.
        ...standard LoRA hyperparams...
        hf_token:    HuggingFace access token. Falls back to HF_TOKEN env var.

    Returns:
        TrainingReport — serialised to spec.output_dir/training_report.json.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ── HF auth ───────────────────────────────────────────────────────────
    token = hf_token or os.getenv("HF_TOKEN", "")
    if not token:
        log.error(
            "HF_TOKEN not set. meta-llama/Llama-3.1-8B-Instruct is gated. "
            "Export: export HF_TOKEN=hf_..."
        )
        sys.exit(1)

    from huggingface_hub import login as hf_login
    hf_login(token=token, add_to_git_credential=False)
    log.info(f"[{spec.task_name}] HF authenticated — base model: {base_model}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log.info(f"[{spec.task_name}] Device: {device} | {gpu_name}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, token=token, padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ────────────────────────────────────────────────────────
    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else torch.float16
    log.info(f"[{spec.task_name}] Loading base model {load_dtype} ...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=token, torch_dtype=load_dtype,
        device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False

    # ── Optimum-AMD ───────────────────────────────────────────────────────
    model, optimum_applied = _apply_optimum_amd(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total if total > 0 else 0.0
    log.info(
        f"[{spec.task_name}] LoRA: trainable={trainable:,}/{total:,} ({pct:.2f}%) "
        f"r={lora_r} alpha={lora_alpha}"
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = build_hf_dataset(spec.examples, spec.prompt_template, eval_frac, seed)
    log.info(
        f"[{spec.task_name}] Dataset: {len(dataset['train'])} train / "
        f"{len(dataset['test'])} eval"
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True,
            max_length=max_seq_len, padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text", "target_json"])
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Training arguments ────────────────────────────────────────────────
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(spec.output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        learning_rate=lr,
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
        report_to="none",
        remove_unused_columns=True,
        dataloader_pin_memory=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )
    log.info(f"[{spec.task_name}] Starting LoRA fine-tuning ...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = round(time.perf_counter() - t0, 1)
    log.info(f"[{spec.task_name}] Training complete in {elapsed}s")

    # ── Save adapter ──────────────────────────────────────────────────────
    model.save_pretrained(str(spec.output_dir))
    tokenizer.save_pretrained(str(spec.output_dir))
    log.info(f"[{spec.task_name}] LoRA adapter saved to {spec.output_dir}/")

    # ── Exact-match eval ──────────────────────────────────────────────────
    model = _reverse_optimum_amd(model, optimum_applied)
    eval_examples_raw = [
        (row["input_text"], json.loads(row["target_json"]))
        for row in dataset["test"]
    ]
    log.info(f"[{spec.task_name}] Computing exact-match on eval set ...")
    eval_metrics = compute_exact_match(
        model, tokenizer, eval_examples_raw,
        spec.prompt_template, spec.output_schema_keys, device,
    )
    log.info(
        f"[{spec.task_name}] Eval exact-match: {eval_metrics['exact_match']:.1%} "
        f"schema-compliance: {eval_metrics['schema_compliance']:.1%}"
    )

    # ── Report ────────────────────────────────────────────────────────────
    train_metrics = train_result.metrics
    report = TrainingReport(
        task_name=spec.task_name,
        base_model=base_model,
        adapter_path=str(spec.output_dir.resolve()),
        dataset_size=len(spec.examples),
        eval_size=len(eval_examples_raw),
        lora_config={
            "r": lora_r, "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        training={
            "epochs": epochs, "max_steps": max_steps,
            "batch_size": batch_size, "grad_accum": grad_accum,
            "lr": lr, "elapsed_s": elapsed,
            "train_loss": round(train_metrics.get("train_loss", -1), 4),
            "train_runtime_s": round(train_metrics.get("train_runtime", 0), 1),
        },
        eval={
            "exact_match":       eval_metrics["exact_match"],
            "schema_compliance": eval_metrics["schema_compliance"],
            "n_examples":        eval_metrics["n_examples"],
            "per_key_accuracy":  eval_metrics["per_key_accuracy"],
        },
        optimum_applied=optimum_applied,
        dtype="bfloat16" if use_bf16 else "float16",
        hardware=f"AMD Instinct MI300X · ROCm" if device.type == "cuda" else "CPU",
        prompt_template=spec.prompt_template,
    )
    report.save(spec.output_dir / "training_report.json")
    return report
