# Big Five Paraphrase Generation and Filtering

Section 6.1 and Appendix A.6 use semantically equivalent paraphrases of the
five Big Five persona prompts to test whether the observed robustness shifts
are stable to surface-form changes.

## Candidate generation

For each trait, the paper generated **20 candidate paraphrases** using
`gpt-4o-2024-08-06`. The generation prompt required each candidate to:

1. begin with "You are..." or "You feel...";
2. preserve the exact number of sentences;
3. preserve sentence order and the corresponding emotional theme;
4. use close synonyms or semantically equivalent expressions without changing
   sentence meaning;
5. remain within **±10%** of the reference prompt word count;
6. form one coherent paragraph; and
7. avoid introducing traits from other Big Five dimensions.

The release provides this generation procedure in
[`scripts/generate_paraphrase_candidates.py`](../scripts/generate_paraphrase_candidates.py).

## Filtering

For each trait, candidates were compared against the original reference prompt
using BERTScore. The paper used the `bert-score` Python package with the
English default **RoBERTa-large** configuration and no baseline rescaling.

A candidate is retained when:

- BERTScore F1 ≥ **0.9**; and
- its word count remains within **±10%** of the reference prompt.

The filtering code is provided in
[`scripts/filter_paraphrase_candidates.py`](../scripts/filter_paraphrase_candidates.py).

The experiment yielded **10 retained paraphrases per trait**. Their full texts
are provided in [`big_five_paraphrases.md`](big_five_paraphrases.md).

Because hosted API model availability can change over time, regenerating the
20 candidates may not produce byte-identical candidate texts. The retained
prompts used in the paper are therefore released directly for exact
reproduction of the downstream experiments.
