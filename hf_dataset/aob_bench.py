"""
hf_dataset/aob_bench.py
========================
HuggingFace Datasets loader for AOB-Bench ClinicalEval v1.

This file is the dataset loading script. When placed in a HuggingFace Space
repository alongside clinical_eval_cases.json, it makes the dataset
loadable via:

    from datasets import load_dataset
    ds = load_dataset("aob-bench/ClinicalEval", split="test")

Local usage (development):
    from datasets import load_dataset
    ds = load_dataset("./aob/hf_dataset", split="test")
"""

import json
import pathlib
from typing import Iterator

import datasets

_DESCRIPTION = """
AOB-Bench ClinicalEval v1 — The first open benchmark for multi-agent
oncology clinical reasoning.

100 expert-curated cases spanning lung adenocarcinoma, lung squamous cell
carcinoma, colon adenocarcinoma, and benign tissue. Each case includes a
pathology report text, patient metadata, and four ground-truth labels:
  - TNM staging (T, N, M, overall stage)
  - Required biomarker panel
  - First-line treatment class (NCCN-aligned)
  - NCCN guideline category
"""

_HOMEPAGE = "https://huggingface.co/datasets/aob-bench/ClinicalEval"
_LICENSE  = "cc-by-4.0"

_CITATION = """\
@software{aob_bench_2026,
  title  = {AOB-Bench: ClinicalEval v1},
  author = {AOB Team},
  year   = {2026},
  url    = {https://huggingface.co/datasets/aob-bench/ClinicalEval},
  note   = {AMD Developer Hackathon 2026. Hardware: AMD Instinct MI300X 192GB HBM3.}
}
"""

_DATA_URL = "clinical_eval_cases.json"


class AobBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for AOB-Bench."""

    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class AobBench(datasets.GeneratorBasedBuilder):
    """AOB-Bench ClinicalEval v1 dataset."""

    BUILDER_CONFIGS = [
        AobBenchConfig(
            name="default",
            description="100 curated oncology clinical reasoning cases.",
        )
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=datasets.Features({
                "case_id":               datasets.Value("string"),
                "pathology_text":        datasets.Value("string"),
                "age":                   datasets.Value("int32"),
                "sex":                   datasets.Value("string"),
                "smoking_history":       datasets.Value("string"),
                "ecog_ps":               datasets.Value("int32"),
                # Ground truth — TNM
                "gt_T":                  datasets.Value("string"),
                "gt_N":                  datasets.Value("string"),
                "gt_M":                  datasets.Value("string"),
                "gt_stage":              datasets.Value("string"),
                # Ground truth — biomarkers + treatment
                "gt_biomarkers":         datasets.Sequence(datasets.Value("string")),
                "gt_first_line_tx_class":datasets.Value("string"),
                "gt_nccn_category":      datasets.Value("string"),
            }),
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        data_file = dl_manager.download_and_extract(_DATA_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_file},
            )
        ]

    def _generate_examples(self, filepath: str) -> Iterator[tuple[int, dict]]:
        with open(filepath, encoding="utf-8") as f:
            cases = json.load(f)

        for idx, case in enumerate(cases):
            meta = case.get("metadata", {})
            gt   = case.get("ground_truth", {})
            tnm  = gt.get("tnm", {})

            yield idx, {
                "case_id":                case.get("case_id", f"AOB-{idx+1:03d}"),
                "pathology_text":         case.get("pathology_text", ""),
                "age":                    int(meta.get("age", -1)),
                "sex":                    meta.get("sex", ""),
                "smoking_history":        meta.get("smoking_history", ""),
                "ecog_ps":                int(meta.get("ecog_ps", -1)),
                "gt_T":                   str(tnm.get("T", "")),
                "gt_N":                   str(tnm.get("N", "")),
                "gt_M":                   str(tnm.get("M", "")),
                "gt_stage":               str(tnm.get("stage", "")),
                "gt_biomarkers":          [str(b) for b in gt.get("biomarkers", [])],
                "gt_first_line_tx_class": str(gt.get("first_line_tx_class", "")),
                "gt_nccn_category":       str(gt.get("nccn_category", "")),
            }
