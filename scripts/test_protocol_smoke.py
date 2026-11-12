"""Protocol-level smoke tests that do not require model weights or API keys."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

import common
from model_loader import TargetLM
from personality_prompts import get_persona_prompts


class _FakeModelWrapper:
    tokenizer = None


def build_llama2_prompt(attack_prompt: str, state_prompt: str) -> str:
    target = TargetLM.__new__(TargetLM)
    target.model_name = "llama2-7b"
    target.model = _FakeModelWrapper()
    target.sys_mode = "replace"
    target.safe_prompt = ""

    conv = common.conv_template("llama-2")
    return target._build_full_prompts(
        [attack_prompt],
        state_prompt,
        [conv],
    )[0]


def main():
    artifact = "FIXED_ATTACK_ARTIFACT_FOR_PROTOCOL_TEST"
    states = get_persona_prompts(mode="full")

    rendered = {
        state: build_llama2_prompt(artifact, system_prompt)
        for state, system_prompt in states.items()
    }

    # The attack artifact must be byte-identical in every rendered prompt.
    for state, prompt in rendered.items():
        count = prompt.count(artifact)
        if count != 1:
            raise AssertionError(
                f"{state}: fixed artifact should appear exactly once; found {count}"
            )

    # Vanilla must not accidentally contain one of the persona prompts.
    vanilla = rendered["vanilla"]
    for state, system_prompt in states.items():
        if state == "vanilla":
            continue
        if system_prompt in vanilla:
            raise AssertionError(
                f"vanilla prompt unexpectedly contains {state} system context"
            )

    # Each non-vanilla state must include its own system context.
    for state, system_prompt in states.items():
        if state == "vanilla":
            continue
        if system_prompt not in rendered[state]:
            raise AssertionError(
                f"{state}: system prompt is missing from rendered target prompt"
            )

    # Prompts should actually differ across operational states.
    if len(set(rendered.values())) != len(rendered):
        raise AssertionError("operational states did not produce distinct prompts")

    # The Llama-2 chat structure should remain valid.
    for state, prompt in rendered.items():
        if not prompt.startswith("<s>[INST]"):
            raise AssertionError(
                f"{state}: unexpected Llama-2 prompt prefix: {prompt[:40]!r}"
            )
        if "[/INST]" not in prompt:
            raise AssertionError(f"{state}: missing Llama-2 assistant boundary")

    print("[protocol-smoke] PASS")
    print(f"  operational states checked: {len(rendered)}")
    print("  fixed attack artifact: unchanged across states")
    print("  system context: state-specific")
    print("  Llama-2 prompt structure: valid")


if __name__ == "__main__":
    main()
