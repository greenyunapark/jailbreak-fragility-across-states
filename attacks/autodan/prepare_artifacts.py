"""Convert official AutoDAN HGA JSON output into the common artifact CSV.

The official AutoDAN implementation stores one record per query with fields
including goal, target, and final_suffix. This script keeps the optimized
suffix fixed so it can be re-evaluated across operational states.

Example:
    python attacks/autodan/prepare_artifacts.py \
        --input_json /path/to/AutoDAN/results/autodan_hga/llama2_0_normal.json \
        --output_csv results/autodan/llama2-7b_artifacts.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for key, item in payload.items():
        if "final_suffix" not in item:
            raise ValueError(f"Record {key} is missing 'final_suffix'.")
        goal = str(item.get("goal", ""))
        suffix = str(item["final_suffix"])
        rows.append({
            "index": int(key) if str(key).isdigit() else key,
            "goal": goal,
            "target": str(item.get("target", "")),
            "suffix": suffix,
            "attack_prompt": suffix.replace("[REPLACE]", goal.lower()),
            "vanilla_generation_success": bool(item.get("is_success", False)),
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} AutoDAN artifacts: {out}")


if __name__ == "__main__":
    main()
