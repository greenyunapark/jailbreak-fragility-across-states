# AutoDAN Adapter

This directory provides the operational-state adapter for **AutoDAN**
(Liu et al.), the white-box jailbreak attack used in the paper.

Official implementation:
https://github.com/SheltonLiu-N/AutoDAN

The upstream repository is MIT licensed. Its license is preserved at
`licenses/AUTODAN_LICENSE`.

## Paper setting

The paper follows the official AutoDAN hyperparameters without modification.
The main protocol is:

1. optimize the AutoDAN artifact under the **vanilla** operational state;
2. keep that optimized artifact fixed;
3. change only the target model's system prompt;
4. evaluate the same artifact under vanilla + the five Big Five states;
5. score responses with the common GPT-4 judge.

Target-model decoding in the paper uses temperature 0.

## 1. Generate AutoDAN artifacts with the official implementation

Clone the official AutoDAN repository and run its HGA implementation with the
official settings and the desired target model. Use the same AdvBench-50
queries released in this repository.

The upstream entry point is:

```bash
python autodan_hga_eval.py ...
```

Because the upstream script uses repository-local model paths, model placement
or path configuration may need to follow the official AutoDAN README.

## 2. Convert the official JSON output

```bash
python attacks/autodan/prepare_artifacts.py \
  --input_json /path/to/AutoDAN/results/autodan_hga/<result>.json \
  --output_csv results/autodan/llama2-7b_artifacts.csv
```

The converter preserves `final_suffix` and materializes the exact target user
prompt after AutoDAN's `[REPLACE]` substitution.

## 3. Re-evaluate the fixed artifact across operational states

```bash
python attacks/autodan/evaluate_operational_states.py \
  --target_model llama2-7b \
  --artifact_csv results/autodan/llama2-7b_artifacts.csv \
  --results_dir results/autodan/llama2-7b
```

## 4. Judge responses

```bash
python attacks/judge_responses.py \
  --responses_csv <autodan-state-output.csv> \
  --judge_model gpt-4-0613
```

### Why generation remains upstream

AutoDAN's HGA optimization is already maintained in the official repository
and the paper explicitly follows its official hyperparameters. This release
therefore avoids vendoring a second copy of the optimizer; it releases the
paper-specific fixed-artifact operational-state evaluation layer instead.
