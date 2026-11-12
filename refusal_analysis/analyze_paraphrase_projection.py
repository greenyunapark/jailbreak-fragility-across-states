"""
Step 3: Analyze paraphrase-level projection onto the refusal direction.

Examines whether semantically equivalent paraphrases of the same persona
cluster in similar positions along the refusal direction.

Usage:
    python analyze_paraphrase_projection.py \
        --npz_path        probe_data/hidden_vectors.npz \
        --directions_path probe_results/probe_directions.npz \
        --output_dir      probe_results
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, f_oneway


TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
LAYER_KEYS = {"early": "X_early", "middle": "X_middle", "late": "X_late"}
TRAIT_COLORS = {
    "Openness":          "#4C72B0",
    "Conscientiousness": "#DD8452",
    "Extraversion":      "#55A868",
    "Agreeableness":     "#C44E52",
    "Neuroticism":       "#8172B2",
}


def project(X, direction, scaler_mean, scaler_scale):
    X_sc = (X - scaler_mean) / scaler_scale
    return X_sc @ direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path",        type=str, required=True,
                        help="Path to hidden_vectors.npz from extract_hidden_states.py.")
    parser.add_argument("--directions_path", type=str, required=True,
                        help="Path to probe_directions.npz from train_probe.py.")
    parser.add_argument("--output_dir",      type=str, default="probe_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    dirs = np.load(args.directions_path, allow_pickle=True)

    personas    = data["personas"]
    prompt_nums = data["prompt_nums"]
    y           = data["y"].astype(int)

    print(f"[data] {len(y)} samples")
    print(f"  prompt_nums: {np.unique(prompt_nums)}")

    sem_mask = prompt_nums != "original"
    print(f"  paraphrase samples: {sem_mask.sum()}")

    layer        = "late"
    X_all        = data[LAYER_KEYS[layer]]
    direction    = dirs[f"direction_{layer}"]
    scaler_mean  = dirs[f"scaler_mean_{layer}"]
    scaler_scale = dirs[f"scaler_scale_{layer}"]

    records = []
    for trait in TRAITS:
        available_prompts = sorted(
            {
                str(p)
                for p in prompt_nums[personas == trait]
                if str(p) != "original"
            },
            key=lambda x: int(x.replace("prompt", "")) if x.startswith("prompt") else x,
        )
        for pnum in available_prompts:
            mask = (personas == trait) & (prompt_nums == pnum)
            projs = project(X_all[mask], direction, scaler_mean, scaler_scale)
            n_succ = y[mask].sum()
            asr = n_succ / mask.sum() * 100
            records.append({
                "trait":     trait,
                "prompt_num": pnum,
                "proj_mean": projs.mean(),
                "proj_std":  projs.std(),
                "asr":       asr,
                "n":         mask.sum(),
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "paraphrase_projection.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[save] {csv_path}")

    print("\n-- Per-trait paraphrase projection stats (late layer) --")
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        print(f"  {trait:20s}: mean={sub['proj_mean'].mean():+.3f}, "
              f"std={sub['proj_mean'].std():.3f}, "
              f"range=[{sub['proj_mean'].min():+.2f}, {sub['proj_mean'].max():+.2f}]  "
              f"ASR mean={sub['asr'].mean():.1f}%")

    projection_groups = [df[df["trait"] == t]["proj_mean"].values for t in TRAITS]
    f_proj, p_proj = f_oneway(*projection_groups)
    print(
        f"\n  Projection ANOVA across traits (paraphrases only): "
        f"F={f_proj:.3f}, p={p_proj:.3g}"
    )

    trait_means = df.groupby("trait")["proj_mean"].mean()
    original_asr = {}
    for trait in TRAITS:
        mask = (personas == trait) & (prompt_nums == "original")
        if mask.sum() > 0:
            original_asr[trait] = float(y[mask].mean() * 100.0)

    missing = [t for t in trait_means.index if t not in original_asr]
    if missing:
        raise ValueError(f"Missing original-prompt labels for: {missing}")

    # Section 6.1 ANOVA: original prompt + 10 retained paraphrases per trait.
    # This yields 55 prompt-level ASR observations and df=(4, 50).
    asr_groups = []
    for trait in TRAITS:
        paraphrase_asr = df[df["trait"] == trait]["asr"].to_numpy(dtype=float)
        asr_groups.append(np.concatenate(([original_asr[trait]], paraphrase_asr)))

    f_asr, p_asr = f_oneway(*asr_groups)
    all_asr = np.concatenate(asr_groups)
    grand_mean = all_asr.mean()
    ss_between = sum(
        len(group) * (group.mean() - grand_mean) ** 2 for group in asr_groups
    )
    ss_total = ((all_asr - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    df_between = len(TRAITS) - 1
    df_within = len(all_asr) - len(TRAITS)
    print(
        f"\n  ASR ANOVA (original + 10 paraphrases per trait): "
        f"F({df_between}, {df_within})={f_asr:.3f}, "
        f"p={p_asr:.3g}, eta^2={eta_sq:.3f}"
    )

    stats_path = os.path.join(args.output_dir, "paraphrase_statistics.csv")
    pd.DataFrame([
        {
            "analysis": "projection_anova_paraphrases",
            "F": f_proj,
            "df_between": len(TRAITS) - 1,
            "df_within": len(df) - len(TRAITS),
            "p": p_proj,
            "eta_squared": np.nan,
        },
        {
            "analysis": "asr_anova_original_plus_paraphrases",
            "F": f_asr,
            "df_between": df_between,
            "df_within": df_within,
            "p": p_asr,
            "eta_squared": eta_sq,
        },
    ]).to_csv(stats_path, index=False)
    print(f"[save] {stats_path}")

    asr_vals = [original_asr[t] for t in trait_means.index]
    r_p, p_p    = pearsonr(trait_means.values, asr_vals)
    r_s, p_s    = spearmanr(trait_means.values, asr_vals)
    print(f"\n  Trait mean projection vs ASR:")
    print(f"    Pearson  r={r_p:+.3f} (p={p_p:.3f})")
    print(f"    Spearman r={r_s:+.3f} (p={p_s:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    positions = np.arange(len(TRAITS))
    for i, trait in enumerate(TRAITS):
        sub_vals = df[df["trait"] == trait]["proj_mean"].values
        ax.boxplot(sub_vals, positions=[i], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=TRAIT_COLORS[trait], alpha=0.7),
                   medianprops=dict(color="black", lw=2))
        orig_mask = (personas == trait) & (prompt_nums == "original")
        if orig_mask.sum() > 0:
            orig_proj = project(X_all[orig_mask], direction,
                                scaler_mean, scaler_scale).mean()
            ax.scatter(i, orig_proj, marker="D", color=TRAIT_COLORS[trait],
                       s=100, zorder=5, edgecolors="black", lw=1.5,
                       label="Original" if i == 0 else "")

    ax.set_xticks(positions)
    ax.set_xticklabels([t[:3] for t in TRAITS], fontsize=11)
    ax.set_ylabel("Projection onto Refusal Direction", fontsize=12)
    ax.set_title("(a) Paraphrase Projection Distribution by Trait", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax.legend(fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(positions, [original_asr[t] for t in TRAITS],
             "D--", color="crimson", lw=1.5, ms=7, label="ASR (%)")
    ax2.set_ylabel("ASR (%)", color="crimson", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="crimson")

    ax = axes[1]
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["proj_mean"], sub["asr"],
                   color=TRAIT_COLORS[trait], s=70, alpha=0.8,
                   label=trait[:3], zorder=5)
    for trait in TRAITS:
        sub = df[df["trait"] == trait]
        ax.scatter(sub["proj_mean"].mean(), original_asr[trait],
                   color=TRAIT_COLORS[trait], s=150, marker="D",
                   edgecolors="black", lw=1.5, zorder=6)

    ax.set_xlabel("Projection onto Refusal Direction (paraphrase mean)", fontsize=12)
    ax.set_ylabel("ASR (%)", fontsize=12)
    ax.set_title("(b) Paraphrase Projection vs ASR", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", lw=0.5)
    ax.legend(title="Trait", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "paraphrase_projection.pdf")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"[plot] Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
