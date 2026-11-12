# Attack Pipelines

The paper evaluates three jailbreak attacks spanning different threat models:

| Attack | Access setting | Release path |
| --- | --- | --- |
| PAIR | black-box | `attacks/pair/` |
| LAA | gray-box | `attacks/generate_suffixes.py` + `attacks/evaluate_operational_states.py` |
| AutoDAN | white-box | `attacks/autodan/` |

All three release paths implement the same experimental separation used in the
paper:

1. generate the attack artifact under the **vanilla** operational state;
2. freeze the generated artifact;
3. vary only the system-prompt-induced operational state;
4. evaluate vanilla + the five Big Five states;
5. score the resulting responses with `attacks/judge_responses.py`.

## LAA

The LAA implementation is adapted and cleaned directly in this repository.

- generate: `generate_suffixes.py`
- state evaluation: `evaluate_operational_states.py`

## PAIR

The PAIR release adapter is in [`pair/`](pair/). It uses the official PAIR
attacker prompts and the paper-reported attacker model
`gpt-3.5-turbo-1106`.

## AutoDAN

The AutoDAN adapter is in [`autodan/`](autodan/). Attack optimization remains
in the official AutoDAN repository; this release converts the resulting
optimized artifact and performs the paper-specific fixed-artifact state
evaluation.

See the top-level [README](../README.md) for Quick Start commands.
