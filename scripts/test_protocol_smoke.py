"""Protocol-level smoke tests that do not require model weights or API keys."""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

import common
from model_loader import TargetLM
from personality_prompts import get_persona_prompts
import evaluate_operational_states as laa_eval


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


class _FakeTargetLM:
    def __init__(self):
        self.calls = []

    def get_response_final(
        self,
        prompts_list,
        persona_prompt=None,
        max_n_tokens=None,
        temperature=None,
        no_template=False,
    ):
        self.calls.append({
            "prompts": list(prompts_list),
            "persona_prompt": persona_prompt,
            "temperature": temperature,
        })
        return [{
            "text": f"synthetic response for state={persona_prompt[:20]}",
            "n_input_tokens": 0,
            "n_output_tokens": 0,
            "logprobs": [],
        } for _ in prompts_list]


def run_stage2_smoke():
    fake_model = _FakeTargetLM()
    original_loader = laa_eval.load_target_model
    laa_eval.load_target_model = lambda _args: fake_model

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            suffix_csv = tmp / "artifact.csv"
            pd.DataFrame([{
                "index": 0,
                "goal": "synthetic evaluation request",
                "suffix": "FIXED_SUFFIX",
            }]).to_csv(suffix_csv, index=False)

            output_csv = laa_eval.run_operational_state_eval(
                target_model="llama2-7b",
                suffix_csv=str(suffix_csv),
                results_dir=str(tmp / "outputs"),
                sys_mode="replace",
                temperature=0.0,
                limit=1,
            )

            out = pd.read_csv(output_csv)
            expected_states = {
                "vanilla",
                "Openness",
                "Conscientiousness",
                "Extraversion",
                "Agreeableness",
                "Neuroticism",
            }
            if len(out) != 6:
                raise AssertionError(
                    f"Stage 2 smoke test expected 6 rows; found {len(out)}"
                )
            if set(out["state"]) != expected_states:
                raise AssertionError(
                    f"Stage 2 produced unexpected states: {set(out['state'])}"
                )
            if out["suffix"].nunique() != 1 or out["suffix"].iloc[0] != "FIXED_SUFFIX":
                raise AssertionError(
                    "Stage 2 did not preserve the fixed jailbreak artifact"
                )
            if len(fake_model.calls) != 6:
                raise AssertionError(
                    f"Expected 6 target-model calls; found {len(fake_model.calls)}"
                )

            rendered_attack_inputs = [
                call["prompts"][0] for call in fake_model.calls
            ]
            if len(set(rendered_attack_inputs)) != 1:
                raise AssertionError(
                    "Attack user input changed across operational states"
                )

            print("[stage2-smoke] PASS")
            print("  one fixed LAA artifact -> six operational-state rows")
            print("  attack input: unchanged across states")
            print("  output CSV schema/state loop: OK")
    finally:
        laa_eval.load_target_model = original_loader


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

    run_stage2_smoke()


if __name__ == "__main__":
    main()
