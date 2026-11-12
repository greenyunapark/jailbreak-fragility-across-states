"""Smoke tests for the PAIR and AutoDAN public adapters."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

from pair.generate_artifacts import extract_json
from pair.upstream_system_prompts import get_attacker_system_prompts


def test_pair_adapter():
    goal = "synthetic harmful-behavior placeholder"
    target = "synthetic target prefix"

    prompts = get_attacker_system_prompts(goal, target)
    if len(prompts) != 3:
        raise AssertionError(
            f"PAIR should expose 3 official attacker strategies; got {len(prompts)}"
        )
    for idx, prompt in enumerate(prompts):
        if goal not in prompt or target not in prompt:
            raise AssertionError(
                f"PAIR system prompt {idx} does not contain goal/target placeholders"
            )

    parsed = extract_json(
        '{"improvement": "try a different framing", '
        '"prompt": "fixed synthetic adversarial prompt"}'
    )
    if parsed is None:
        raise AssertionError("PAIR JSON parser rejected valid attacker output")
    if parsed["prompt"] != "fixed synthetic adversarial prompt":
        raise AssertionError("PAIR JSON parser changed the attack prompt")

    print("[pair-adapter-smoke] PASS")
    print("  official attacker strategies: wired")
    print("  JSON attack artifact parsing: OK")


def test_autodan_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_json = tmp / "autodan.json"
        output_csv = tmp / "artifacts.csv"

        input_json.write_text(
            """{
  "0": {
    "goal": "Synthetic Goal",
    "target": "Synthetic Target",
    "final_suffix": "PREFIX [REPLACE] SUFFIX",
    "is_success": true
  }
}""",
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "attacks/autodan/prepare_artifacts.py"),
                "--input_json",
                str(input_json),
                "--output_csv",
                str(output_csv),
            ],
            check=True,
        )

        df = pd.read_csv(output_csv)
        if len(df) != 1:
            raise AssertionError(
                f"AutoDAN converter should emit 1 row; found {len(df)}"
            )
        row = df.iloc[0]
        if row["suffix"] != "PREFIX [REPLACE] SUFFIX":
            raise AssertionError("AutoDAN converter did not preserve final_suffix")
        if row["attack_prompt"] != "PREFIX synthetic goal SUFFIX":
            raise AssertionError(
                "AutoDAN converter did not match official [REPLACE] substitution"
            )

    print("[autodan-adapter-smoke] PASS")
    print("  final_suffix: preserved")
    print("  [REPLACE] -> instruction.lower(): OK")


def main():
    test_pair_adapter()
    test_autodan_adapter()


if __name__ == "__main__":
    main()
