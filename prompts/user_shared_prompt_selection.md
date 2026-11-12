# User-Shared Prompt Selection

The 15 system prompts used in Section 6.3 were sourced from the
[fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)
community prompt-sharing dataset.

The purpose of this set is **cross-prompt-family validation**. It is not
intended to represent an exhaustive or statistically representative sample of
deployment-time system prompts.

## Inclusion criteria

A prompt was retained only if it:

1. begins with **"Act as"**, indicating a role assignment suitable for
   system-level conditioning; and
2. contains **50–136 words**, corresponding to ±30% of the word-count range of
   the Big Five persona prompts (47–105 words).

## Exclusion criteria

Prompts were excluded if they:

- contain user-turn indicators such as "My first request is" or
  "I will provide a";
- contain template variables such as `${variable}`;
- reference specific external resources, files, uploaded documents, or other
  context that must be supplied separately;
- are written in, or explicitly require output in, a non-English language;
- reference specific brand names, proprietary tools, or real-world entities in
  ways that introduce uncontrolled semantic variation;
- are designed for image or video generation rather than text interaction; or
- contain harmful or borderline content such as gambling or explicit
  lifestyle content.

Applying these criteria produced a **51-prompt candidate pool**.

## Sampling

The final 15 prompts were sampled from that candidate pool with a fixed random
seed:

```python
selected = candidate_pool.sample(n=15, random_state=42)
```

The same sampling step is provided in
[`scripts/sample_user_shared_prompts.py`](../scripts/sample_user_shared_prompts.py).

The final selected prompts are released in:

- [`datasets/user_shared_prompts.csv`](../datasets/user_shared_prompts.csv)
- [`prompts/user_shared_prompts.md`](user_shared_prompts.md)

The selected prompt labels are:

1. Philosophy Tutor
2. Creative Writing Coach
3. RPG Game Master
4. Executive Assistant
5. Wildlife Narrator
6. QA Automation Engineer
7. Resume Editor
8. Lead Data Analyst
9. English Tutor
10. Virtualization Expert
11. EdTech Developer
12. Backend Engineer
13. Data Analyst
14. Frontend Developer
15. Performance Engineer
