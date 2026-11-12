"""Evaluate fixed LAA suffixes under retained Big Five paraphrases.

This script reproduces the response-generation stage of the Section 6.1
paraphrase analysis. The jailbreak suffix is held fixed while the system prompt
is changed to each retained paraphrase.

Example:
    python attacks/evaluate_paraphrases.py \
        --target_model llama2-13b \
        --suffix_csv results/laa/results_llama2-13b_advbench_50/LAA_llama2-13b_replace.csv \
        --results_dir results/paraphrases/llama2-13b \
        --temperature 0.0
"""

import argparse
import datetime
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REFUSAL_DIR = ROOT / "refusal_analysis"
if str(REFUSAL_DIR) not in sys.path:
    sys.path.insert(0, str(REFUSAL_DIR))

import personality_prompts_semantic as semantic_prompts

from judges import judge_rule_based
from model_loader import load_target_model
from prompts import get_universal_manual_prompt_sys
from utils import insert_adv_string

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]


def build_base_prompt(goal: str, prompt_template: str) -> str:
    return get_universal_manual_prompt_sys(
        prompt_template,
        "",
        goal.lower(),
    )


def run_trait(
    target_lm,
    suffix_df,
    trait: str,
    prompts: dict,
    results_dir: str,
    target_model: str,
    prompt_template: str,
    target_max_n_tokens: int,
    temperature: float,
):
    rows = []

    for prompt_id, system_prompt in prompts.items():
        print(f"\n===== {trait} / {prompt_id} =====")
        for _, row in tqdm(
            suffix_df.iterrows(),
            total=len(suffix_df),
            desc=f"{trait}:{prompt_id}",
            leave=False,
        ):
            try:
                goal = str(row["goal"])
                suffix = str(row["suffix"])
                attack_prompt = insert_adv_string(
                    build_base_prompt(goal, prompt_template),
                    suffix,
                )
                response = target_lm.get_response_final(
                    persona_prompt=system_prompt,
                    prompts_list=[attack_prompt],
                    max_n_tokens=target_max_n_tokens,
                    temperature=temperature,
                )[0]["text"]

                rows.append({
                    "index": int(row["index"]),
                    "goal": goal,
                    "suffix": suffix,
                    "state": trait,
                    "prompt_num": prompt_id,
                    "temperature": temperature,
                    "response": response,
                    "rule_based_success": bool(judge_rule_based(response)),
                })
            except Exception as exc:
                print(
                    f"[ERROR] trait={trait} prompt={prompt_id} "
                    f"index={row.get('index', 'N/A')}: {exc}"
                )
                traceback.print_exc()

    trait_dir = Path(results_dir) / trait
    trait_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = trait_dir / (
        f"temp{temperature}_{target_model}_{trait.lower()}_{timestamp}.csv"
    )
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="llama2-13b")
    parser.add_argument("--suffix_csv", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/paraphrases")
    parser.add_argument(
        "--sys_mode",
        type=str,
        default="replace",
        choices=["append", "prepend", "replace"],
    )
    parser.add_argument("--prompt_template", type=str, default="best_llama2")
    parser.add_argument("--target_max_n_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    suffix_df = pd.read_csv(args.suffix_csv, encoding="utf-8-sig")
    required = {"index", "goal", "suffix"}
    missing = required.difference(suffix_df.columns)
    if missing:
        raise ValueError(
            f"Suffix CSV is missing required columns: {sorted(missing)}"
        )

    class _Args:
        target_model = args.target_model
        persona_prompt = ""
        sys_mode = args.sys_mode
        safe_prompt = ""

    target_lm = load_target_model(_Args())

    for trait in TRAITS:
        prompts = getattr(semantic_prompts, trait)
        run_trait(
            target_lm=target_lm,
            suffix_df=suffix_df,
            trait=trait,
            prompts=prompts,
            results_dir=args.results_dir,
            target_model=args.target_model,
            prompt_template=args.prompt_template,
            target_max_n_tokens=args.target_max_n_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
