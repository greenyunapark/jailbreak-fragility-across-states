# PAIR Adapter

This directory provides a thin release adapter for **PAIR** (Chao et al.),
the black-box jailbreak attack used in the paper.

Upstream implementation:
https://github.com/patrickrchao/JailbreakingLLMs

The upstream PAIR repository is MIT licensed. Its license is preserved at
`licenses/PAIR_LICENSE`.

## Paper setting

The paper follows the official PAIR configuration and uses:

- attacker model: `gpt-3.5-turbo-1106`
- attack generation under the **vanilla** operational state
- the generated adversarial prompt is then held fixed across operational states
- target decoding temperature: 0
- final ASR judge: `gpt-4-0613`, score 10 = jailbreak success

## Generate vanilla PAIR artifacts

```bash
python attacks/pair/generate_artifacts.py \
  --target_model llama2-7b \
  --csv_path datasets/advbench_50.csv \
  --output_csv results/pair/llama2-7b_pair_artifacts.csv
```

The release adapter serializes the highest-scored PAIR prompt encountered for
each query so it can be reused unchanged across states.

Historical OpenAI model identifiers may no longer be available on every
account. Replacing the attacker or judge model can change the results.

## Evaluate fixed PAIR prompts across states

```bash
python attacks/pair/evaluate_operational_states.py \
  --target_model llama2-7b \
  --artifact_csv results/pair/llama2-7b_pair_artifacts.csv \
  --results_dir results/pair/llama2-7b
```

Then run the common GPT judge:

```bash
python attacks/judge_responses.py \
  --responses_csv <pair-state-output.csv> \
  --judge_model gpt-4-0613
```

The PAIR attacker system prompts in `upstream_system_prompts.py` are copied
from the official implementation and retain the upstream MIT license.
