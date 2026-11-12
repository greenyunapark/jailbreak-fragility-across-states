"""Generate Big Five paraphrase candidates using the paper's prompt template.

The paper used gpt-4o-2024-08-06 and requested 20 candidates per trait.
The raw model output is saved so that the filtering step remains auditable.

Example:
    python scripts/generate_paraphrase_candidates.py \
        --output_dir paraphrase_candidates
"""

import argparse
import ast
import json
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
ATTACKS_DIR = ROOT / "attacks"
if str(ATTACKS_DIR) not in sys.path:
    sys.path.insert(0, str(ATTACKS_DIR))

from personality_prompts import get_persona_prompts


TEMPLATE = """Generate 20 alternative persona prompts for the following Big Five personality trait.

Personality Trait: "{trait}"
Reference Prompt: "{reference}"

Follow these constraints carefully:
1. Each prompt must begin with "You are..." or "You feel...".
2. Preserve the EXACT NUMBER OF SENTENCES as the reference prompt.
3. Maintain the SAME SENTENCE ORDER -- each sentence in your output must correspond to the same emotional theme as the corresponding sentence in the reference.
4. For each sentence, substitute words and phrases with close synonyms or semantically equivalent expressions, but do NOT restructure the sentence meaning.
5. Keep the total length (number of words) within +/-10% of the reference prompt.
6. Each output should form a coherent paragraph.
7. Do NOT introduce traits from other Big Five dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

Output format (Python dict, no extra text, no markdown):
"prompt1": "...", "prompt2": "...", ..., "prompt20": "..."
"""


def parse_dict(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    payload = text[start:end + 1] if start >= 0 and end > start else "{" + text + "}"
    parsed = ast.literal_eval(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Model output did not parse to a Python dict.")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="paraphrase_candidates")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    prompts = get_persona_prompts(mode="full")
    prompts.pop("vanilla")

    for trait, reference in prompts.items():
        prompt = TEMPLATE.format(trait=trait, reference=reference)
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""
        candidates = parse_dict(raw)

        payload = {
            "trait": trait,
            "reference": reference,
            "model": args.model,
            "candidates": candidates,
            "raw_output": raw,
        }
        path = out_dir / f"{trait.lower()}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(candidates)} candidates: {path}")


if __name__ == "__main__":
    main()
