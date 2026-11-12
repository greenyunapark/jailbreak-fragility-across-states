"""Evaluate fixed PAIR adversarial prompts across operational states."""

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

from judges import judge_rule_based
from model_loader import load_target_model
from personality_prompts import get_persona_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--artifact_csv", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/pair/eval")
    parser.add_argument("--target_max_n_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.artifact_csv, encoding="utf-8-sig")
    required = {"index", "goal", "attack_prompt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"PAIR artifact CSV is missing required columns: {sorted(missing)}"
        )

    class _Args:
        target_model = args.target_model
        persona_prompt = ""
        sys_mode = "replace"
        safe_prompt = ""

    target_lm = load_target_model(_Args())
    states = get_persona_prompts(mode="full")

    rows = []
    for state_name, system_prompt in states.items():
        print(f"\n===== Operational state: {state_name} =====")
        prompts = df["attack_prompt"].astype(str).tolist()
        responses = target_lm.get_response_final(
            persona_prompt=system_prompt,
            prompts_list=prompts,
            max_n_tokens=args.target_max_n_tokens,
            temperature=args.temperature,
        )
        for (_, row), response in zip(df.iterrows(), responses):
            text = response["text"]
            rows.append({
                "index": int(row["index"]),
                "goal": str(row["goal"]),
                "target": str(row.get("target", "")),
                "attack_prompt": str(row["attack_prompt"]),
                "state": state_name,
                "temperature": args.temperature,
                "response": text,
                "rule_based_success": bool(judge_rule_based(text)),
            })

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.target_model}_pair_states_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved responses: {out_path}")
    print("Score with attacks/judge_responses.py for the paper's GPT-4 ASR.")


if __name__ == "__main__":
    main()
