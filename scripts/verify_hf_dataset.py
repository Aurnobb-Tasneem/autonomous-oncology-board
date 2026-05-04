"""Verify the HF dataset structure."""
import json, pathlib

cases = json.loads(pathlib.Path("aob/hf_dataset/clinical_eval_cases.json").read_text())
print(f"Dataset: {len(cases)} cases")
print(f"First case id: {cases[0]['case_id']}")
print(f"Last case id: {cases[-1]['case_id']}")

cancer_types = {}
for c in cases:
    stage = c.get("ground_truth", {}).get("tnm", {}).get("stage", "")
    if stage == "benign" or "benign" in stage:
        cancer_types["benign"] = cancer_types.get("benign", 0) + 1
    elif "squamous" in c["pathology_text"].lower():
        cancer_types["lung_squamous"] = cancer_types.get("lung_squamous", 0) + 1
    elif "colon" in c["pathology_text"].lower() or "rectal" in c["pathology_text"].lower():
        cancer_types["colon_adeno"] = cancer_types.get("colon_adeno", 0) + 1
    else:
        cancer_types["lung_adeno"] = cancer_types.get("lung_adeno", 0) + 1
print("Distribution:", cancer_types)

# Verify schema
required_fields = ["case_id", "pathology_text", "metadata", "ground_truth"]
gt_required = ["tnm", "biomarkers", "first_line_tx_class", "nccn_category"]
for c in cases:
    for f in required_fields:
        assert f in c, f"Missing field {f} in {c.get('case_id')}"
    for f in gt_required:
        assert f in c["ground_truth"], f"Missing GT field {f} in {c.get('case_id')}"
print("Schema validation: PASSED")
