# Dataset Notes

This directory contains evaluation artifacts used in the paper.

| File | Source / role | License / terms |
| --- | --- | --- |
| `advbench_50.csv` | 50-query subset of AdvBench used in the main jailbreak evaluation | AdvBench / `llm-attacks` MIT license |
| `advbench_520_classified.csv` | AdvBench-520 with the harm-category annotations used in the scope analysis | Original AdvBench content under MIT; added annotations released as research artifacts with this repository |
| `user_shared_prompts.csv` | 15 community-shared system prompts used in Section 6.3 | Source prompt collection is CC0 1.0 |

## AdvBench

AdvBench originates from the repository accompanying:

**Universal and Transferable Adversarial Attacks on Aligned Language Models**  
Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson.

Source:
https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench

The upstream repository is distributed under the MIT License. The preserved
license notice is included at [`licenses/ADVBENCH_LICENSE`](../licenses/ADVBENCH_LICENSE).

## User-shared system prompts

The prompts were selected from the community prompt-sharing dataset used in
the paper:

https://huggingface.co/datasets/fka/awesome-chatgpt-prompts

The source collection is released under **CC0 1.0 Universal**. The exact
selection criteria and sampling procedure used in the paper are documented in
[`prompts/user_shared_prompt_selection.md`](../prompts/user_shared_prompt_selection.md).

When reusing benchmark data, please also cite the corresponding original
dataset or paper.
