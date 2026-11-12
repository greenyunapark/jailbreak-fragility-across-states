# Third-Party Code

## LLM Adaptive Attacks (LAA)

Parts of the jailbreak attack implementation in `attacks/` are adapted from:

**Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**  
Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion.  
ICLR 2025.

Original repository:
https://github.com/tml-epfl/llm-adaptive-attacks

The original code is distributed under the MIT License.

We modify the original implementation to support the operational-state
evaluation setting used in our paper, including system-prompt conditioning,
persona-conditioned evaluation, and evaluation across multiple operational
states.

The original LAA repository is itself partially based on the PAIR repository:
https://github.com/patrickrchao/JailbreakingLLMs


The upstream MIT license text is preserved in
[`licenses/LAA_LICENSE`](licenses/LAA_LICENSE).


## AdvBench

The files `datasets/advbench_50.csv` and the original query content in
`datasets/advbench_520_classified.csv` are derived from AdvBench, released
with the `llm-attacks` repository:

https://github.com/llm-attacks/llm-attacks

Copyright (c) 2023 Andy Zou. The upstream repository is distributed under the
MIT License. See `licenses/ADVBENCH_LICENSE`.

## Awesome ChatGPT Prompts

The user-shared system prompts in `datasets/user_shared_prompts.csv` and
`prompts/user_shared_prompts.md` were selected from the community dataset
used in the paper:

https://huggingface.co/datasets/fka/awesome-chatgpt-prompts

The source prompt collection is released under CC0 1.0 Universal.


## PAIR

The PAIR release adapter in `attacks/pair/` follows the official
JailbreakingLLMs implementation:

https://github.com/patrickrchao/JailbreakingLLMs

The file `attacks/pair/upstream_system_prompts.py` is copied from the
official PAIR repository for reproducibility.

Copyright (c) 2023 PAIR Team. The upstream project is distributed under the
MIT License. See `licenses/PAIR_LICENSE`.

## AutoDAN

The AutoDAN operational-state adapter in `attacks/autodan/` is designed to
consume attack artifacts produced by the official AutoDAN implementation:

https://github.com/SheltonLiu-N/AutoDAN

The AutoDAN optimizer itself is not vendored in this repository.

Copyright (c) 2023 Xiaogeng Liu. The upstream project is distributed under the
MIT License. See `licenses/AUTODAN_LICENSE`.
