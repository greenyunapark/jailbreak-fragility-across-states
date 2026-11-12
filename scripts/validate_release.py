"""Static release-usability checks for the public repository.

This script intentionally uses only the Python standard library so it can run
before installing the research dependencies.
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".env.example",
    "datasets/advbench_50.csv",
    "attacks/generate_suffixes.py",
    "attacks/evaluate_operational_states.py",
    "attacks/judge_responses.py",
    "attacks/pair/generate_artifacts.py",
    "attacks/pair/evaluate_operational_states.py",
    "attacks/autodan/prepare_artifacts.py",
    "attacks/autodan/evaluate_operational_states.py",
]

REQUIRED_REQUIREMENTS = {
    "torch",
    "transformers",
    "accelerate",
    "fschat",
    "openai",
    "anthropic",
    "tiktoken",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "sentencepiece",
    "bert-score",
}

REQUIRED_ENV_KEYS = {"OPENAI_API_KEY", "HF_TOKEN"}

CLI_CONTRACTS = {
    "attacks/generate_suffixes.py": {
        "--target_model",
        "--csv_path",
        "--results_dir",
        "--sys_mode",
        "--limit",
    },
    "attacks/evaluate_operational_states.py": {
        "--target_model",
        "--suffix_csv",
        "--results_dir",
        "--sys_mode",
        "--temperature",
        "--limit",
    },
    "attacks/judge_responses.py": {
        "--responses_csv",
        "--judge_model",
    },
    "attacks/pair/generate_artifacts.py": {
        "--target_model",
        "--csv_path",
        "--output_csv",
    },
    "attacks/pair/evaluate_operational_states.py": {
        "--target_model",
        "--artifact_csv",
        "--results_dir",
    },
    "attacks/autodan/prepare_artifacts.py": {
        "--input_json",
        "--output_csv",
    },
    "attacks/autodan/evaluate_operational_states.py": {
        "--target_model",
        "--artifact_csv",
        "--results_dir",
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"[release-check] FAIL: {message}")


def parse_requirement_name(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        if sep in line:
            line = line.split(sep, 1)[0]
    return line.strip().lower()


def argparse_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


def main() -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            fail(f"required release file is missing: {rel}")

    requirements = {
        name
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (name := parse_requirement_name(line))
    }
    missing_requirements = REQUIRED_REQUIREMENTS - requirements
    if missing_requirements:
        fail(
            "requirements.txt is missing: "
            + ", ".join(sorted(missing_requirements))
        )

    env_keys = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            env_keys.add(line.split("=", 1)[0].strip())
    missing_env = REQUIRED_ENV_KEYS - env_keys
    if missing_env:
        fail(".env.example is missing: " + ", ".join(sorted(missing_env)))

    with (ROOT / "datasets/advbench_50.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        if not {"goal", "target"}.issubset(columns):
            fail("datasets/advbench_50.csv must contain goal,target columns")
        rows = list(reader)
        if len(rows) != 50:
            fail(
                f"datasets/advbench_50.csv should contain 50 rows; found {len(rows)}"
            )

    for rel, expected_flags in CLI_CONTRACTS.items():
        flags = argparse_flags(ROOT / rel)
        missing = expected_flags - flags
        if missing:
            fail(
                f"{rel} is missing README-facing CLI options: "
                + ", ".join(sorted(missing))
            )

    # Pure persona mode must truly mean vanilla-only generation.
    namespace: dict[str, object] = {}
    exec(
        (ROOT / "attacks/personality_prompts.py").read_text(encoding="utf-8"),
        namespace,
    )
    get_persona_prompts = namespace["get_persona_prompts"]
    pure = get_persona_prompts("pure")
    full = get_persona_prompts("full")
    if pure != {"vanilla": ""}:
        fail(f"pure persona mode is not vanilla-only: {pure!r}")
    expected_states = {
        "vanilla",
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    }
    if set(full) != expected_states:
        fail(f"full operational-state set is unexpected: {set(full)!r}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Requirements",
        "## Installation",
        "## Quick Start",
        "## Running Experiments",
    ):
        if heading not in readme:
            fail(f"README is missing heading: {heading}")

    for token in (
        "pip install -r requirements.txt",
        "python attacks/generate_suffixes.py",
        "python attacks/evaluate_operational_states.py",
        "python attacks/pair/generate_artifacts.py",
        "python attacks/autodan/prepare_artifacts.py",
    ):
        if token not in readme:
            fail(f"README is missing release command: {token}")

    print("[release-check] PASS")
    print("  files: OK")
    print("  requirements: OK")
    print("  environment-variable template: OK")
    print("  AdvBench-50 schema/row count: OK")
    print("  CLI contracts: OK")
    print("  operational-state prompt sets: OK")
    print("  README release sections/commands: OK")


if __name__ == "__main__":
    main()
