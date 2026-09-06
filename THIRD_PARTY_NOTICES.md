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
