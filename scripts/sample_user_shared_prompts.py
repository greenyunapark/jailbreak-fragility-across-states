"""Sample the 15 user-shared prompts used in the paper.

The paper first applies the documented inclusion/exclusion criteria to the
source dataset, producing a 51-prompt candidate pool. This script reproduces
the final random sampling step from that candidate pool.

Expected input columns:
    name, prompt

Example:
    python scripts/sample_user_shared_prompts.py \
        --candidate_csv candidate_prompts_51.csv \
        --output_csv selected_prompts_15.csv
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.candidate_csv, encoding="utf-8-sig")
    required = {"name", "prompt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Candidate CSV is missing required columns: {sorted(missing)}"
        )
    if len(df) < args.n:
        raise ValueError(
            f"Requested {args.n} prompts but candidate pool has only {len(df)} rows."
        )

    sampled = df.sample(n=args.n, random_state=args.random_state)
    sampled.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(
        f"Sampled {len(sampled)} prompts from {len(df)} candidates "
        f"(random_state={args.random_state})."
    )
    print(sampled["name"].to_string(index=False))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
