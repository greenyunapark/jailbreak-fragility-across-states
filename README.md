# The Fragility of Jailbreak Robustness Across Operational States

Official code and experimental artifacts for:

**The Fragility of Jailbreak Robustness Across Operational States**  
*Findings of EMNLP 2026*

**Yuna Park, Hwang Youn Kim, Yujin Kim, Won Woo Ro, Suhyun Kim, Jae-In Hwang**

[📄 arXiv](https://arxiv.org/abs/2608.30748)

---

<p align="center">
  <a href="intro_figure.pdf">
    <img src="intro_figure.png" width="900"
         alt="Illustration of state-induced robustness shift">
  </a>
</p>



## Overview

Jailbreak robustness is commonly characterized using a single attack success
rate (ASR) measured under a default, or *vanilla*, model configuration.

In this work, we show that jailbreak robustness can vary substantially when
only the model's **context-induced operational state** is changed, while the
target model and jailbreak artifact remain fixed. We refer to this phenomenon
as **state-induced robustness shift**.

We systematically study this phenomenon across seven aligned language models
and three representative jailbreak attacks. In some settings, changing only
the operational state changes ASR by as much as **56 percentage points
(2% → 58%)**, without modifying the jailbreak artifact itself.

We further show that state-dependent robustness variation is systematically
associated with differences in hidden representations along a
**refusal-related axis**, providing a predictive representation-level account
of the observed robustness shifts.

This repository provides the code and experimental artifacts associated with
our study, including operational-state prompts, LAA-based evaluation code,
and refusal-related representation analysis.

---

## Key Idea

Our evaluation separates **attack generation** from **operational-state
evaluation**:

1. A jailbreak artifact is generated under the vanilla operational state.
2. The target model and jailbreak artifact are kept fixed.
3. Only the system prompt used to induce the operational state is changed.
4. The same artifact is evaluated across different operational states.
5. Attack success rates are compared across states.

This setup allows us to examine whether a robustness measurement obtained in
one operational state remains stable when the model's context changes.

---

## Repository Structure

```text
.
├── attacks/
│   ├── generate_suffixes.py
│   ├── evaluate_personas.py
│   ├── laa_attack.py
│   ├── model_loader.py
│   ├── language_models.py
│   ├── prompts.py
│   ├── personality_prompts.py
│   ├── judges.py
│   ├── config.py
│   └── utils.py
│
├── datasets/
│   └── advbench_520_classified.csv
│
├── prompts/
│   ├── big_five_paraphrases.md
│   └── user_shared_prompts.md
│
├── refusal_analysis/
│   ├── extract_hidden_states.py
│   ├── train_probe.py
│   ├── analyze_paraphrase_projection.py
│   └── personality_prompts_semantic.py
│
├── intro_figure.pdf
├── intro_figure.png
├── THIRD_PARTY_NOTICES.md
├── .gitignore
└── README.md
```

---

## Released Artifacts

### Operational-State Prompts

The prompts used to instantiate and analyze operational states are provided in
[`prompts/`](prompts/).

Currently released prompt artifacts include:

- [`big_five_paraphrases.md`](prompts/big_five_paraphrases.md)  
  Paraphrased variants of the five Big Five persona prompts used in the
  paraphrase robustness analysis.

- [`user_shared_prompts.md`](prompts/user_shared_prompts.md)  
  User-shared role prompts used to evaluate whether state-induced robustness
  shifts extend beyond the controlled Big Five prompt family.

The original Big Five persona prompts used for state conditioning are also
included in
[`attacks/personality_prompts.py`](attacks/personality_prompts.py).

The Big Five prompts are used as a systematic and reproducible instrument for
inducing controlled state variation. They are **not** intended to represent a
canonical or comprehensive distribution of real-world system prompts.

---

## LAA-Based Operational-State Evaluation

The currently released jailbreak evaluation code focuses on experiments based
on **LLM Adaptive Attacks (LAA)**.

The original LAA implementation is adapted so that a jailbreak artifact can be
generated under the vanilla state and subsequently evaluated across different
system-prompt-induced operational states.

The main scripts are:

### 1. Jailbreak Artifact Generation

[`attacks/generate_suffixes.py`](attacks/generate_suffixes.py)

Generates jailbreak artifacts using the LAA attack.

In the main experimental setting of the paper, jailbreak artifacts are
generated under the vanilla operational state.

### 2. State-Conditioned Evaluation

[`attacks/evaluate_personas.py`](attacks/evaluate_personas.py)

Evaluates pre-generated jailbreak artifacts across different operational
states while keeping the target model and attack artifact fixed.

### 3. Operational-State Prompts

[`attacks/personality_prompts.py`](attacks/personality_prompts.py)

Contains the Big Five persona prompts used to induce the five controlled
non-vanilla operational states:

- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

### 4. Model and Evaluation Utilities

The remaining files in [`attacks/`](attacks/) provide model loading,
prompt formatting, attack logic, judging, and shared utilities.

---

## Models and Attacks in the Paper

The paper evaluates seven aligned language models:

- Llama-2-7B-Chat
- Llama-2-13B-Chat
- Llama-3-8B-Instruct
- Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct
- Vicuna-7B-v1.5

and three representative jailbreak attacks:

- **PAIR** — black-box
- **LAA** — gray-box
- **AutoDAN** — white-box

The code currently organized in this repository focuses on the
**LAA-based experimental pipeline**.

For PAIR and AutoDAN, please refer to their original implementations:

- [PAIR](https://github.com/patrickrchao/JailbreakingLLMs)
- [LAA](https://github.com/tml-epfl/llm-adaptive-attacks)
- [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN)

---

## Representation-Level Analysis

The [`refusal_analysis/`](refusal_analysis/) directory contains the code used
for the representation-level analyses in the paper.

We investigate whether operational-state variation is associated with
systematic changes in hidden representations along a refusal-related axis.

The analysis pipeline consists of three main stages.

### 1. Hidden-State Extraction

[`refusal_analysis/extract_hidden_states.py`](refusal_analysis/extract_hidden_states.py)

Extracts hidden representations immediately before response generation across
different operational states.

### 2. Refusal-Related Probe

[`refusal_analysis/train_probe.py`](refusal_analysis/train_probe.py)

Trains a logistic-regression probe to distinguish jailbreak success and
failure from hidden representations.

The learned weight vector is used as a **refusal-related direction**.

### 3. Projection Analysis

[`refusal_analysis/analyze_paraphrase_projection.py`](refusal_analysis/analyze_paraphrase_projection.py)

Analyzes how representations induced by different operational-state prompts
project onto the learned refusal-related direction, including the paraphrase
analysis reported in the paper.

Our representation analysis should be interpreted as a **predictive and
correlational account** of state-dependent jailbreak robustness, rather than
as evidence of a causal mechanism.

---

## Datasets

The [`datasets/`](datasets/) directory contains data used in additional
analyses reported in the paper.

The repository currently includes:

- `advbench_520_classified.csv` — AdvBench queries with harm-category
  annotations used for the scope analysis.

The main experiments in the paper use the AdvBench evaluation subset together
with additional evaluations on MaliciousInstruct and JailbreakBench.

Please refer to the original dataset releases for their respective licenses
and terms of use.

---

## Dependencies

The codebase is written in Python and uses libraries including:

- PyTorch
- Hugging Face Transformers
- FastChat
- OpenAI
- pandas
- NumPy
- scikit-learn
- tqdm
- BERTScore

Some evaluated models require access approval and a Hugging Face token.

A version-pinned environment specification and verified end-to-end
reproduction commands will be added as the public research code is further
organized.

---

## Third-Party Code

Parts of the jailbreak attack implementation in this repository are adapted
from:

**Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**  
Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion.

Original repository:  
https://github.com/tml-epfl/llm-adaptive-attacks

The LAA implementation is distributed under the MIT License. The original LAA
repository itself contains code partially based on the PAIR implementation.

Our modifications support the operational-state evaluation setting studied in
this work, including system-prompt conditioning and evaluation of fixed
jailbreak artifacts across multiple operational states.

See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
for attribution and licensing information.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{park2026fragility,
  title   = {The Fragility of Jailbreak Robustness Across Operational States},
  author  = {Park, Yuna and Kim, Hwang Youn and Kim, Yujin and Ro, Won Woo
             and Kim, Suhyun and Hwang, Jae-In},
  journal = {arXiv preprint arXiv:2608.30748},
  year    = {2026}
}
```

The paper was accepted to **Findings of EMNLP 2026**. The official ACL
Anthology citation will be added after publication.

---

## Acknowledgements

We thank the authors of the open-source attack implementations and datasets
used in this work.

For detailed attribution of reused or adapted code, please see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
