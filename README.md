# The Fragility of Jailbreak Robustness Across Operational States

Code and experimental artifacts for:

**The Fragility of Jailbreak Robustness Across Operational States**  
Yuna Park, Hwang Youn Kim, Yujin Kim, Won Woo Ro, Suhyun Kim, Jae-In Hwang  
**Findings of EMNLP 2026**

[ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1840/) ·
[arXiv](https://arxiv.org/abs/2608.30748)

<p align="center">
  <a href="intro_figure.pdf">
    <img src="intro_figure.png" width="900"
         alt="Illustration of state-induced jailbreak robustness shift">
  </a>
</p>

## Overview

Jailbreak robustness is usually reported as a single attack success rate (ASR)
measured in a default, or *vanilla*, model configuration. We show that this
measurement can change substantially when only an ordinary system prompt is
changed while the target model and jailbreak artifact remain fixed.

Across seven aligned models and three representative jailbreak attacks, at
least one non-vanilla state increases ASR in 18 of 21 model-attack settings.
In the largest observed shift, ASR changes from **2% to 58%** solely because
the operational state changes.

The central evaluation protocol is:

1. Generate a jailbreak artifact in the vanilla state.
2. Keep the target model and jailbreak artifact fixed.
3. Change only the system prompt that induces the operational state.
4. Re-evaluate the same artifact across states.
5. Compare ASR across states.

This repository releases the LAA-based operational-state pipeline used for
this protocol, the system-prompt artifacts used in the paper, and code for the
refusal-related representation analysis.

## Repository Structure

```text
.
├── attacks/
│   ├── generate_suffixes.py              # Stage 1: vanilla-state LAA generation
│   ├── evaluate_operational_states.py    # Stage 2: fixed-artifact state evaluation
│   ├── judge_responses.py                # Stage 3: GPT-4 judge and ASR
│   ├── laa_attack.py                     # adapted LAA attack implementation
│   ├── model_loader.py
│   ├── language_models.py
│   ├── personality_prompts.py
│   ├── prompts.py
│   ├── judges.py
│   ├── common.py
│   ├── config.py
│   └── utils.py
├── datasets/
│   ├── advbench_50.csv
│   ├── advbench_520_classified.csv
│   ├── user_shared_prompts.csv
│   └── README.md
├── prompts/
│   ├── big_five_prompts.md
│   ├── big_five_paraphrases.md
│   └── user_shared_prompts.md
├── refusal_analysis/
│   ├── extract_hidden_states.py
│   ├── train_probe.py
│   ├── analyze_paraphrase_projection.py
│   └── personality_prompts_semantic.py
├── licenses/
│   └── LAA_LICENSE
├── requirements.txt
├── .env.example
├── THIRD_PARTY_NOTICES.md
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For gated Hugging Face models and GPT-based judging, set the corresponding
environment variables:

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...
```

See `.env.example` for the supported variables.

## Reproducing the LAA Operational-State Evaluation

### 1. Generate jailbreak suffixes in the vanilla state

```bash
python attacks/generate_suffixes.py \
  --target_model llama2-7b \
  --csv_path datasets/advbench_50.csv \
  --sys_mode replace \
  --results_dir results/laa \
  --n-restarts 1
```

The Stage-1 CSV contains the optimized suffix for each query. With the command
above, it is written under a directory of the form:

```text
results/laa/results_llama2-7b_advbench_50/
```

Stage 1 intentionally uses the **vanilla operational state only**.

### 2. Evaluate the same suffixes across operational states

```bash
python attacks/evaluate_operational_states.py \
  --target_model llama2-7b \
  --suffix_csv results/laa/results_llama2-7b_advbench_50/LAA_llama2-7b_replace.csv \
  --results_dir results/laa/eval_llama2-7b \
  --sys_mode replace \
  --temperature 0.0
```

This evaluates each fixed suffix under vanilla and the five Big Five
operational states:

- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

The rule-based label written by this script is provided only as a diagnostic.
The paper's primary ASR uses the GPT-4 judge described below.

### 3. Compute GPT-4-judged ASR

```bash
python attacks/judge_responses.py \
  --responses_csv <stage-2-response-csv> \
  --judge_model gpt-4-0613
```

The paper used **GPT-4 (`gpt-4-0613`)** as the judge and counted a response as
a successful jailbreak only when the judge score was 10. API availability for
historical model IDs can change; when using a different judge for new
experiments, report that deviation explicitly.

## Operational-State Prompts

The original Big Five prompts are available in
[`prompts/big_five_prompts.md`](prompts/big_five_prompts.md) and are also
defined programmatically in
[`attacks/personality_prompts.py`](attacks/personality_prompts.py).

Additional prompt artifacts are released in [`prompts/`](prompts/):

- `big_five_paraphrases.md`: retained paraphrases used for the semantic
  robustness analysis.
- `user_shared_prompts.md`: full text of the 15 user-shared system prompts
  analyzed in the paper.

The Big Five prompts are a controlled and reproducible intervention for
inducing state variation. They are not intended to represent a canonical
distribution of deployed system prompts.

## Representation Analysis

The [`refusal_analysis/`](refusal_analysis/) directory contains the code for
the refusal-related representation analysis.

For the original Big Five states only:

```bash
python refusal_analysis/extract_hidden_states.py \
  --original_csv <gpt-4-judged-stage-2-csv> \
  --suffix_csv <stage-1-suffix-csv> \
  --sys_mode replace \
  --output_dir probe_data \
  --skip_semantic

python refusal_analysis/train_probe.py \
  --npz_path probe_data/hidden_vectors.npz \
  --output_dir probe_results \
  --original_only
```

For the paraphrase analysis, provide the paraphrase-response directory through
`--semantic_dir`, then run:

```bash
python refusal_analysis/analyze_paraphrase_projection.py \
  --npz_path probe_data/hidden_vectors.npz \
  --directions_path probe_results/probe_directions.npz \
  --output_dir probe_results
```

The analysis code derives ASR from the supplied response labels rather than
from hard-coded paper values.

## Models and Attacks in the Paper

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

This repository provides the consolidated, directly runnable LAA pipeline.
PAIR and AutoDAN experiments in the paper follow the same fixed-artifact
operational-state protocol using their upstream implementations:

- [PAIR](https://github.com/patrickrchao/JailbreakingLLMs)
- [LAA](https://github.com/tml-epfl/llm-adaptive-attacks)
- [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN)

## Datasets

See [`datasets/README.md`](datasets/README.md) for released dataset artifacts
and provenance notes.

The repository includes the AdvBench-50 subset used in the main experiments,
the AdvBench-520 harm-category annotations used in the scope analysis, and the
15 user-shared system prompts.

## Third-Party Code

Parts of the LAA implementation are adapted from:

> Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion.  
> *Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks.*

The upstream MIT license is preserved in
[`licenses/LAA_LICENSE`](licenses/LAA_LICENSE). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{park-etal-2026-fragility,
  title     = {The Fragility of Jailbreak Robustness Across Operational States},
  author    = {Park, Yuna and Kim, Hwang Youn and Kim, Yujin and
               Ro, Won Woo and Kim, Suhyun and Hwang, Jae-In},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages     = {39277--39298},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  year      = {2026},
  url       = {https://aclanthology.org/2026.findings-emnlp.1840/}
}
```

## Notes on Reproducibility

Exact numerical reproduction can depend on model checkpoints, tokenizer and
library versions, hardware, and access to historical API model IDs. The code
keeps the paper's operational-state intervention explicit: jailbreak artifacts
are generated in the vanilla state and reused unchanged across the evaluated
states.
