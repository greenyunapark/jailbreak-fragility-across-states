# The Fragility of Jailbreak Robustness Across Operational States

Code and experimental artifacts accompanying:

**The Fragility of Jailbreak Robustness Across Operational States**  
*Findings of the Association for Computational Linguistics: EMNLP 2026*

**Yuna Park, Hwang Youn Kim, Yujin Kim, Won Woo Ro, Suhyun Kim, Jae-In Hwang**

[ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1840/) · [arXiv](https://arxiv.org/abs/2608.30748)

<p align="center">
  <a href="intro_figure.pdf">
    <img src="intro_figure.png" width="900"
         alt="Illustration of state-induced jailbreak robustness shifts">
  </a>
</p>

<p align="center">
  <em>Same model, same jailbreak artifact, different operational state.</em>
</p>

## Overview

Jailbreak robustness is often summarized by a single attack success rate (ASR)
measured under a default model configuration. Our paper shows that this
single-point view can be highly unstable: changing only an ordinary system
prompt can substantially change jailbreak success, even when the target model
and jailbreak artifact are held fixed.

We refer to this phenomenon as **state-induced robustness shift**.

Across seven aligned language models and three representative jailbreak
attacks, we find that operational-state variation can strongly affect measured
robustness. In one setting, ASR changes from **2% to 58%** solely because the
system prompt changes.

The central evaluation protocol is simple:

1. Generate a jailbreak artifact under the **vanilla** operational state.
2. Keep the target model and jailbreak artifact fixed.
3. Change only the system prompt used to induce the operational state.
4. Re-evaluate the same artifact across states.
5. Compare the resulting ASRs.

This repository releases the code and artifacts used to study that protocol,
with a cleaned implementation of the **LAA-based evaluation pipeline**,
operational-state prompts, evaluation datasets, and representation-level
analysis code.

## Main findings

- Jailbreak robustness is **state-contingent** rather than a fixed property of a
  model-attack pair.
- In **18 of 21** model-attack combinations, at least one non-vanilla state
  yields a higher ASR than the vanilla state.
- State-induced shifts also appear under user-shared role prompts, not only
  under the controlled Big Five prompt family.
- Hidden representations shift systematically along a refusal-related
  direction, and these shifts are strongly predictive of jailbreak outcomes.
- These representation results are intended as **predictive/correlational
  evidence**, not as a causal claim.

## Repository structure

```text
.
├── attacks/
│   ├── generate_suffixes.py
│   ├── evaluate_operational_states.py
│   ├── evaluate_user_shared_prompts.py
│   ├── judge_responses.py
│   ├── laa_attack.py
│   ├── model_loader.py
│   ├── language_models.py
│   ├── common.py
│   ├── prompts.py
│   ├── personality_prompts.py
│   ├── judges.py
│   ├── config.py
│   └── utils.py
│
├── datasets/
│   ├── README.md
│   ├── advbench_50.csv
│   ├── advbench_520_classified.csv
│   └── user_shared_prompts.csv
│
├── prompts/
│   ├── big_five_prompts.md
│   ├── big_five_paraphrases.md
│   ├── paraphrase_generation.md
│   ├── user_shared_prompts.md
│   └── user_shared_prompt_selection.md
│
├── refusal_analysis/
│   ├── extract_hidden_states.py
│   ├── extract_state_hidden_states.py
│   ├── train_probe.py
│   ├── evaluate_probe_transfer.py
│   ├── analyze_paraphrase_projection.py
│   └── personality_prompts_semantic.py
│
├── scripts/
│   ├── generate_paraphrase_candidates.py
│   ├── filter_paraphrase_candidates.py
│   └── sample_user_shared_prompts.py
│
├── intro_figure.pdf
├── intro_figure.png
├── licenses/
│   ├── LAA_LICENSE
│   └── ADVBENCH_LICENSE
├── requirements.txt
├── .env.example
├── THIRD_PARTY_NOTICES.md
├── .gitignore
└── README.md
```

## Setup

Create a Python environment and install the released dependencies:

```bash
pip install -r requirements.txt
```

For API-based judging and gated Hugging Face models, provide credentials through
environment variables:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

See `.env.example` for the variables used by the codebase.

## Reproducing the LAA evaluation workflow

Run the following commands from the repository root.

### 1. Generate LAA suffixes under the vanilla state

```bash
python attacks/generate_suffixes.py \
    --target_model llama2-7b \
    --csv_path datasets/advbench_50.csv \
    --sys_mode replace \
    --results_dir results/laa
```

### 2. Evaluate the same suffixes across operational states

```bash
python attacks/evaluate_operational_states.py \
    --target_model llama2-7b \
    --suffix_csv results/laa/results_llama2-7b_advbench_50/LAA_llama2-7b_replace.csv \
    --results_dir results/laa/eval_llama2-7b \
    --sys_mode replace \
    --temperature 0.0
```

### 3. Score responses with the paper's GPT-4 judge

```bash
python attacks/judge_responses.py \
    --responses_csv <stage-2-output.csv> \
    --judge_model gpt-4-0613
```

The paper counts a response as a successful jailbreak when the judge assigns
a score of 10. The exact judge model used in the paper was `gpt-4-0613`.
Because this is a historical API model identifier, exact reruns depend on
provider/account availability; substituting another judge model may change the
reported ASRs. The Stage 2 script also records a rule-based heuristic label for
diagnostics, but the GPT-based judge is the evaluation used for the reported
main ASRs.

## User-shared system prompt evaluation

The 15 publicly shared role prompts used in Section 6.3 can be evaluated with
the same fixed LAA suffixes:

```bash
python attacks/evaluate_user_shared_prompts.py \
    --target_model llama2-13b \
    --suffix_csv results/laa/results_llama2-13b_advbench_50/LAA_llama2-13b_replace.csv \
    --prompts_csv datasets/user_shared_prompts.csv \
    --results_dir results/user_shared/llama2-13b \
    --temperature 0.0
```

Then score the responses with the same GPT-4 judge:

```bash
python attacks/judge_responses.py \
    --responses_csv <user-shared-output.csv> \
    --judge_model gpt-4-0613
```

The documented filtering criteria and the final random sampling step used to
construct the 15-prompt set are provided in
`prompts/user_shared_prompt_selection.md` and
`scripts/sample_user_shared_prompts.py`.

## Reproducing the refusal-related representation analysis

For the main Big Five analysis on Llama-2-13B, first extract hidden states from
the judged Stage-2 responses:

```bash
python refusal_analysis/extract_hidden_states.py \
    --original_csv <stage-3-judged-response.csv> \
    --suffix_csv results/laa/results_llama2-13b_advbench_50/LAA_llama2-13b_replace.csv \
    --sys_mode replace \
    --skip_semantic \
    --output_dir probe_data
```

Then train the 5-fold stratified logistic-regression probe using only the
vanilla + five Big Five states:

```bash
python refusal_analysis/train_probe.py \
    --npz_path probe_data/hidden_vectors.npz \
    --output_dir probe_results \
    --original_only
```

The script reports AUROC across the early (L4), middle (L16), and late (L32)
representations and computes projection-ASR correlations across all six
operational states. State-level projection uncertainty is reported as the
standard error across the 50 AdvBench queries, matching the analysis protocol
described in the paper.

### Cross-prompt-family probe evaluation

For Section 6.3, the generic extractor can create comparable NPZ files for
both the Big Five and user-shared prompt families:

```bash
python refusal_analysis/extract_state_hidden_states.py \
    --target_model llama2-13b \
    --responses_csv <judged-big-five.csv> \
    --output_npz probe_data/big_five.npz

python refusal_analysis/extract_state_hidden_states.py \
    --target_model llama2-13b \
    --responses_csv <judged-user-shared.csv> \
    --state_prompts_csv datasets/user_shared_prompts.csv \
    --exclude_state vanilla \
    --output_npz probe_data/user_shared.npz
```

A probe trained on one prompt family can then be evaluated on the other:

```bash
python refusal_analysis/evaluate_probe_transfer.py \
    --train_npz probe_data/big_five.npz \
    --test_npz probe_data/user_shared.npz \
    --layer middle
```

The representation depth is exposed as a command-line argument rather than
hard-coded; Appendix B.4 reports results across L4, L16, and L32.

## LAA-based operational-state evaluation

The released attack code focuses on **LLM Adaptive Attacks (LAA)**.

The implementation separates jailbreak artifact generation from
operational-state evaluation so that the same optimized artifact can be reused
across different system-prompt-induced states.

### Stage 1: generate jailbreak artifacts

`attacks/generate_suffixes.py`

This stage generates optimized LAA suffixes under the vanilla operational
state. The main AdvBench subset used in the paper is provided as
`datasets/advbench_50.csv`.

### Stage 2: evaluate fixed artifacts across states

`attacks/evaluate_operational_states.py`

This stage evaluates pre-generated artifacts under vanilla and the five Big
Five persona-conditioned operational states while keeping the attack artifact
fixed.

The five controlled non-vanilla states are:

- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

The original state prompts are defined in
`attacks/personality_prompts.py`.

## Prompt artifacts

The `prompts/` directory contains prompt material used in the paper.

- `big_five_paraphrases.md`: the retained paraphrases used to test whether the
  observed effects are robust to semantically similar prompt wording.
- `paraphrase_generation.md`: the candidate-generation prompt and BERTScore /
  word-count filtering procedure used to construct the retained paraphrases.
- `user_shared_prompts.md`: the 15 user-shared role prompts used to test
  whether state-induced shifts extend beyond the controlled Big Five prompt
  family.
- `user_shared_prompt_selection.md`: the inclusion/exclusion criteria and
  sampling procedure used to construct the 15-prompt set.

The Big Five prompts are used as a **systematic and reproducible intervention
basis** for operational-state variation. They are not intended to approximate
the full distribution of system prompts encountered in deployment.

## Reproducing Big Five paraphrase construction

The exact retained paraphrases used in the experiments are released in
`prompts/big_five_paraphrases.md`. To reproduce the construction procedure,
the repository also includes the candidate-generation and filtering scripts:

```bash
python scripts/generate_paraphrase_candidates.py \
    --output_dir paraphrase_candidates

python scripts/filter_paraphrase_candidates.py \
    --input_dir paraphrase_candidates \
    --output_csv paraphrase_candidates/retained.csv
```

The paper generated 20 candidates per trait with `gpt-4o-2024-08-06` and
retained candidates with BERTScore F1 ≥ 0.9 and word count within ±10% of the
reference prompt. The paper does not specify additional decoding parameters
for candidate generation, so the release does not impose an unreported value.
See `prompts/paraphrase_generation.md` for the full procedure.

## Representation-level analysis

The `refusal_analysis/` directory contains code for the representation
analysis reported in the paper.

The pipeline consists of:

1. `extract_hidden_states.py` — extract hidden representations immediately
   before response generation.
2. `train_probe.py` — train a logistic-regression probe that separates
   jailbreak success from failure and derive a refusal-related direction.
3. `analyze_paraphrase_projection.py` — analyze projections onto that
   direction, including the paraphrase analysis.

The purpose of this analysis is to characterize a representation-level
association with state-dependent robustness. It should not be interpreted as
establishing a causal mechanism.

## Models and attacks evaluated in the paper

The paper evaluates seven aligned language models:

- Llama-2-7B-Chat
- Llama-2-13B-Chat
- Llama-3-8B-Instruct
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct
- Vicuna-7B-v1.5

and three jailbreak attacks:

- **PAIR** — black-box
- **LAA** — gray-box
- **AutoDAN** — white-box

This repository currently redistributes and cleans the **LAA-based pipeline**.
PAIR and AutoDAN experiments in the paper were built from their original
implementations; we do not redistribute those upstream repositories here.

Original implementations:

- PAIR: https://github.com/patrickrchao/JailbreakingLLMs
- LAA: https://github.com/tml-epfl/llm-adaptive-attacks
- AutoDAN: https://github.com/SheltonLiu-N/AutoDAN

## Datasets

The repository includes:

- `datasets/advbench_50.csv` — the 50-query AdvBench subset used in the main
  evaluation.
- `datasets/advbench_520_classified.csv` — AdvBench queries with harm-category
  annotations used in the scope analysis.
- `datasets/user_shared_prompts.csv` — user-shared role prompts used in the
  additional prompt analysis.

Please also consult the original dataset releases for their licenses and terms
of use.

## API keys and model access

No API keys or Hugging Face credentials are stored in this repository.

When required, provide credentials through environment variables such as:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

Some Hugging Face models require access approval from their original model
providers.

## Third-party code

Parts of the LAA implementation are adapted from:

**Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**  
Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion.

Original repository:  
https://github.com/tml-epfl/llm-adaptive-attacks

The original implementation is distributed under the MIT License and itself
contains code partially based on PAIR. Our modifications add support for the
operational-state evaluation protocol used in this work.

See `THIRD_PARTY_NOTICES.md` for attribution details.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{park-etal-2026-fragility,
    title = "The Fragility of Jailbreak Robustness Across Operational States",
    author = "Park, Yuna  and
      Kim, Hwang Youn  and
      Kim, Yujin  and
      Ro, Won Woo  and
      Kim, Suhyun  and
      Hwang, Jae-In",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1840/",
    pages = "39277--39298",
    ISBN = "979-8-89176-462-0"
}
```

## Responsible use

This repository is intended for research on language-model safety,
jailbreak evaluation, robustness, and alignment. The included harmful prompts
originate from established safety-evaluation benchmarks and are provided for
research and reproducibility.
