"""
Evaluate fixed LAA suffixes under the 15 user-shared system prompts.

This reproduces the LAA prompt-family validation used in Section 6.3:
the target model and jailbreak suffix are held fixed, while only the system
prompt changes. The selected prompts are provided in
datasets/user_shared_prompts.csv.

Example:
    python attacks/evaluate_user_shared_prompts.py \
        --target_model llama2-13b \
        --suffix_csv results/laa/results_llama2-13b_advbench_50/LAA_llama2-13b_replace.csv \
        --prompts_csv datasets/user_shared_prompts.csv \
        --results_dir results/user_shared/llama2-13b \
        --temperature 0.0
"""

import argparse
import datetime
import os
import random
import traceback

import numpy as np
import pandas as pd
import torch

from judges import judge_rule_based
from model_loader import load_target_model
from prompts import get_universal_manual_prompt_sys
from utils import insert_adv_string

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def build_base_prompt(goal: str, target_str: str = "", prompt_template: str = "best_llama2"):
    return get_universal_manual_prompt_sys(prompt_template, target_str, goal.lower())


def run_user_shared_eval(
    target_model: str,
    suffix_csv: str,
    prompts_csv: str,
    results_dir: str,
    sys_mode: str = "replace",
    prompt_template: str = "best_llama2",
    target_max_n_tokens: int = 150,
    temperature: float = 0.0,
    safe_prompt: str = "",
    include_vanilla: bool = True,
):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    suffix_df = pd.read_csv(suffix_csv, encoding="utf-8-sig")
    required_suffix = {"index", "goal", "suffix"}
    missing = required_suffix.difference(suffix_df.columns)
    if missing:
        raise ValueError(
            f"Suffix CSV is missing required columns: {sorted(missing)}. "
            "Use the CSV produced by attacks/generate_suffixes.py."
        )

    prompts_df = pd.read_csv(prompts_csv, encoding="utf-8-sig")
    required_prompts = {"name", "prompt"}
    missing = required_prompts.difference(prompts_df.columns)
    if missing:
        raise ValueError(
            f"Prompt CSV is missing required columns: {sorted(missing)}."
        )

    class _Args:
        pass

    model_args = _Args()
    model_args.target_model = target_model
    model_args.persona_prompt = ""
    model_args.sys_mode = sys_mode
    model_args.safe_prompt = safe_prompt
    target_lm = load_target_model(model_args)

    states = []
    if include_vanilla:
        states.append(("vanilla", ""))
    states.extend(
        (str(row["name"]), str(row["prompt"]))
        for _, row in prompts_df.iterrows()
    )

    rows = []
    print(f"[LOAD] {len(suffix_df)} fixed jailbreak artifacts from {suffix_csv}")
    print(f"[LOAD] {len(states)} operational states ({len(prompts_df)} user-shared + vanilla={include_vanilla})")

    for state_name, state_prompt in states:
        print(f"\n===== Operational state: {state_name} =====")
        for _, row in tqdm(
            suffix_df.iterrows(),
            total=len(suffix_df),
            desc=state_name,
            leave=False,
        ):
            try:
                goal = str(row["goal"])
                suffix = str(row["suffix"])
                base_prompt = build_base_prompt(
                    goal=goal,
                    target_str="",
                    prompt_template=prompt_template,
                )
                attack_prompt = insert_adv_string(base_prompt, suffix)

                output = target_lm.get_response_final(
                    persona_prompt=state_prompt,
                    prompts_list=[attack_prompt],
                    max_n_tokens=target_max_n_tokens,
                    temperature=temperature,
                )[0]["text"]

                rows.append({
                    "index": int(row["index"]),
                    "goal": goal,
                    "suffix": suffix,
                    "state": state_name,
                    "temperature": temperature,
                    "response": output,
                    "rule_based_success": bool(judge_rule_based(output)),
                })
            except Exception as exc:
                print(
                    f"[ERROR] index={row.get('index', 'N/A')} "
                    f"state={state_name}: {exc}"
                )
                traceback.print_exc()

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(
        results_dir,
        f"{target_model}_user_shared_temp{temperature}_{timestamp}.csv",
    )
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved responses: {output_csv}")
    print(
        "For GPT-4-based ASR, run attacks/judge_responses.py on this file."
    )
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--suffix_csv", type=str, required=True)
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="datasets/user_shared_prompts.csv",
    )
    parser.add_argument("--results_dir", type=str, default="results/user_shared")
    parser.add_argument(
        "--sys_mode",
        type=str,
        default="replace",
        choices=["append", "prepend", "replace"],
    )
    parser.add_argument("--prompt_template", type=str, default="best_llama2")
    parser.add_argument("--target_max_n_tokens", type=int, default=150)
    parser.add_argument("--safe_prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no_vanilla",
        action="store_true",
        help="Do not include the vanilla baseline in the output.",
    )
    args = parser.parse_args()

    run_user_shared_eval(
        target_model=args.target_model,
        suffix_csv=args.suffix_csv,
        prompts_csv=args.prompts_csv,
        results_dir=args.results_dir,
        sys_mode=args.sys_mode,
        prompt_template=args.prompt_template,
        target_max_n_tokens=args.target_max_n_tokens,
        temperature=args.temperature,
        safe_prompt=args.safe_prompt,
        include_vanilla=not args.no_vanilla,
    )
