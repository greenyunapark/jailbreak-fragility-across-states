"""Extract hidden states from a judged operational-state response CSV.

This generic extractor supports both the Big Five evaluation and the
user-shared prompt evaluation. It reconstructs the fixed LAA attack input from
(goal, suffix), applies the corresponding system prompt, and extracts the
hidden state of the final input token before generation.

Examples:

Big Five:
    python refusal_analysis/extract_state_hidden_states.py \
        --target_model llama2-13b \
        --responses_csv <judged-big-five.csv> \
        --output_npz probe_data/big_five.npz

User-shared prompts:
    python refusal_analysis/extract_state_hidden_states.py \
        --target_model llama2-13b \
        --responses_csv <judged-user-shared.csv> \
        --state_prompts_csv datasets/user_shared_prompts.csv \
        --output_npz probe_data/user_shared.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

import common
from model_loader import load_target_model
from personality_prompts import get_persona_prompts
from prompts import get_universal_manual_prompt_sys
from utils import insert_adv_string


TARGET_LAYERS = {"early": 4, "middle": 16, "late": 32}


def parse_success(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.astype(int).to_numpy()
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
        .to_numpy()
    )


def load_state_prompts(csv_path: str | None) -> dict[str, str]:
    if csv_path is None:
        return get_persona_prompts(mode="full")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"name", "prompt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"State-prompt CSV is missing required columns: {sorted(missing)}"
        )
    mapping = {"vanilla": ""}
    mapping.update(
        {
            str(row["name"]): str(row["prompt"])
            for _, row in df.iterrows()
        }
    )
    return mapping


def build_full_prompt(target_lm, attack_prompt: str, system_prompt: str) -> str:
    conv = common.conv_template(target_lm.template)
    return target_lm._build_full_prompts(
        [attack_prompt],
        system_prompt,
        [conv],
    )[0]


@torch.no_grad()
def extract_vector(target_lm, full_prompt: str) -> dict[str, np.ndarray]:
    tokenizer = target_lm.model.tokenizer
    model = target_lm.model.model
    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_pos = inputs["attention_mask"].sum(dim=1).item() - 1

    return {
        name: hidden_states[min(layer_idx, len(hidden_states) - 1)][
            0, last_pos, :
        ].float().cpu().numpy()
        for name, layer_idx in TARGET_LAYERS.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="llama2-13b")
    parser.add_argument("--responses_csv", type=str, required=True)
    parser.add_argument(
        "--state_prompts_csv",
        type=str,
        default=None,
        help=(
            "Optional CSV with columns name,prompt. If omitted, vanilla + "
            "Big Five prompts are used."
        ),
    )
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--sys_mode", type=str, default="replace")
    parser.add_argument("--prompt_template", type=str, default="best_llama2")
    parser.add_argument(
        "--exclude_state",
        action="append",
        default=[],
        help="State name to exclude; may be supplied multiple times.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.responses_csv, encoding="utf-8-sig")
    required = {"index", "goal", "suffix", "state", "jailbreak_success"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Judged response CSV is missing required columns: {sorted(missing)}"
        )

    if args.exclude_state:
        df = df[~df["state"].isin(args.exclude_state)].copy()

    labels = parse_success(df["jailbreak_success"])
    state_prompts = load_state_prompts(args.state_prompts_csv)

    unknown = sorted(set(df["state"]) - set(state_prompts))
    if unknown:
        raise ValueError(
            "No system prompt found for states: "
            + ", ".join(map(str, unknown))
        )

    class _Args:
        target_model = args.target_model
        persona_prompt = ""
        sys_mode = args.sys_mode
        safe_prompt = ""

    print(f"[model] Loading {args.target_model}...")
    target_lm = load_target_model(_Args())
    print("[model] Loaded.")

    vectors = {name: [] for name in TARGET_LAYERS}
    states = []
    query_idx = []

    for row_pos, (_, row) in enumerate(df.iterrows(), start=1):
        goal = str(row["goal"])
        suffix = str(row["suffix"])
        state = str(row["state"])

        base_prompt = get_universal_manual_prompt_sys(
            args.prompt_template,
            "",
            goal.lower(),
        )
        attack_prompt = insert_adv_string(base_prompt, suffix)
        full_prompt = build_full_prompt(
            target_lm,
            attack_prompt,
            state_prompts[state],
        )
        extracted = extract_vector(target_lm, full_prompt)

        for name in TARGET_LAYERS:
            vectors[name].append(extracted[name])
        states.append(state)
        query_idx.append(int(row["index"]))

        if row_pos % 25 == 0 or row_pos == len(df):
            print(f"[extract] {row_pos}/{len(df)}")

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_early=np.stack(vectors["early"]),
        X_middle=np.stack(vectors["middle"]),
        X_late=np.stack(vectors["late"]),
        y=labels,
        states=np.asarray(states, dtype=object),
        personas=np.asarray(states, dtype=object),
        query_idx=np.asarray(query_idx),
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
