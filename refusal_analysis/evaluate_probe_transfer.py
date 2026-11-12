"""Train/evaluate a refusal probe across operational-state prompt families.

For cross-family evaluation, fit the logistic-regression probe on train_npz and
evaluate query-level AUROC plus state-level projection-ASR correlation on
test_npz.

For within-family evaluation, pass the same NPZ to --train_npz and --test_npz
and add --cross_validate. AUROC is then estimated with 5-fold stratified CV,
while state-level projection statistics are computed from a probe fit on all
examples.

The input NPZ files can be created with extract_state_hidden_states.py.
"""

import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


LAYER_KEYS = {
    "early": "X_early",
    "middle": "X_middle",
    "late": "X_late",
}


def load_npz(path: str, layer: str, excluded_states: set[str]):
    data = np.load(path, allow_pickle=True)
    X = data[LAYER_KEYS[layer]]
    y = data["y"].astype(int)
    states = data["states"] if "states" in data else data["personas"]

    if excluded_states:
        mask = np.array([str(s) not in excluded_states for s in states])
        X = X[mask]
        y = y[mask]
        states = states[mask]

    return X, y, np.asarray(states, dtype=object)


def fit_probe(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=1)
    probe.fit(X_scaled, y)
    return probe, scaler


def state_statistics(X, y, states, probe, scaler):
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    projections = scaler.transform(X) @ direction

    names = list(dict.fromkeys(map(str, states)))
    rows = []
    for state in names:
        mask = np.asarray(states, dtype=str) == state
        rows.append(
            (
                state,
                float(projections[mask].mean()),
                float(y[mask].mean() * 100.0),
                int(mask.sum()),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument(
        "--layer",
        type=str,
        default="middle",
        choices=list(LAYER_KEYS),
    )
    parser.add_argument(
        "--exclude_state",
        action="append",
        default=[],
        help=(
            "State to exclude from both train and test sets. For the public-"
            "prompt correlation reported in Section 6.3, exclude vanilla."
        ),
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Report 5-fold stratified CV AUROC on the training set.",
    )
    args = parser.parse_args()

    excluded = set(args.exclude_state)
    X_train, y_train, train_states = load_npz(
        args.train_npz, args.layer, excluded
    )
    X_test, y_test, test_states = load_npz(
        args.test_npz, args.layer, excluded
    )

    if args.cross_validate:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        cv_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        scores = cross_val_score(
            cv_probe,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="roc_auc",
        )
        print(
            f"5-fold CV AUROC: {scores.mean():.3f} +/- {scores.std():.3f}"
        )

    probe, scaler = fit_probe(X_train, y_train)
    test_prob = probe.predict_proba(scaler.transform(X_test))[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)
    print(f"Test AUROC: {test_auc:.3f}")

    rows = state_statistics(
        X_test, y_test, test_states, probe, scaler
    )
    if len(rows) < 3:
        raise ValueError("Need at least 3 states for projection-ASR correlation.")

    state_names = [r[0] for r in rows]
    projection_means = [r[1] for r in rows]
    asrs = [r[2] for r in rows]
    pearson_r, pearson_p = pearsonr(projection_means, asrs)
    spearman_r, spearman_p = spearmanr(projection_means, asrs)

    print("\nState-level projection / ASR:")
    for state, projection, asr, n in rows:
        print(
            f"  {state:30s} projection={projection:+.4f} "
            f"ASR={asr:6.2f}% n={n}"
        )

    print(
        f"\nPearson r={pearson_r:.3f} (p={pearson_p:.3g})\n"
        f"Spearman rho={spearman_r:.3f} (p={spearman_p:.3g})"
    )


if __name__ == "__main__":
    main()
