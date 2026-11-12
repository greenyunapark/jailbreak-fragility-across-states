"""Filter Big Five paraphrase candidates using the paper's criteria.

Candidates are retained when:
  1. BERTScore F1 >= 0.9 relative to the reference prompt; and
  2. word count is within +/-10% of the reference prompt.

BERTScore is computed with the bert-score package using the English default
RoBERTa-large configuration and no baseline rescaling, matching Appendix A.6.

Example:
    python scripts/filter_paraphrase_candidates.py \
        --input_dir paraphrase_candidates \
        --output_csv paraphrase_candidates/retained.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from bert_score import score


def word_count(text: str) -> int:
    return len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--length_tolerance", type=float, default=0.10)
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        trait = payload["trait"]
        reference = payload["reference"]
        candidates = payload["candidates"]

        ids = list(candidates)
        texts = [candidates[k] for k in ids]
        references = [reference] * len(texts)

        _, _, f1 = score(
            texts,
            references,
            lang="en",
            verbose=True,
            rescale_with_baseline=False,
        )

        ref_words = word_count(reference)
        lower = ref_words * (1 - args.length_tolerance)
        upper = ref_words * (1 + args.length_tolerance)

        for prompt_id, text, bert_f1 in zip(ids, texts, f1.tolist()):
            n_words = word_count(text)
            length_ok = lower <= n_words <= upper
            bert_ok = bert_f1 >= args.threshold
            records.append({
                "trait": trait,
                "prompt_id": prompt_id,
                "prompt": text,
                "word_count": n_words,
                "reference_word_count": ref_words,
                "bert_score_f1": bert_f1,
                "length_ok": length_ok,
                "bert_score_ok": bert_ok,
                "retained": bool(length_ok and bert_ok),
            })

    df = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    kept = df[df["retained"]]
    print(kept.groupby("trait").size().rename("retained"))
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
