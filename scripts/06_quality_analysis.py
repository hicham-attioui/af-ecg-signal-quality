from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, FIG = ROOT / "data" / "processed", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLASSES = ["N", "A", "O", "~"]
RHYTHMS = ["N", "A", "O"]                     
NAMES = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}
COVERAGES = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
SHORT_SECONDS = 20


def macro_f1(true, pred, labels):
    return f1_score(true, pred, labels=labels, average="macro", zero_division=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# A. Abstention
def abstention(cnn):
    rows = []
    for seed, g in cnn.groupby("seed"):
        g = g.sort_values("confidence", ascending=False)        
        for cov in COVERAGES:
            n = int(round(cov * len(g)))
            kept, refused = g.iloc[:n], g.iloc[n:]
            rand = g.sample(n=n, random_state=seed)              
            row = {"seed": seed, "coverage": cov,
                   "macro_f1": macro_f1(kept["true"], kept["pred"], CLASSES),
                   "challenge_f1": macro_f1(kept["true"], kept["pred"], RHYTHMS),
                   "random_macro_f1": macro_f1(rand["true"], rand["pred"], CLASSES)}
            for c in CLASSES:   
                row[f"refused_{NAMES[c]}"] = (refused["true"] == c).sum() / (g["true"] == c).sum()
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "abstention.csv", index=False)

    mean, std = res.groupby("coverage").mean(), res.groupby("coverage").std()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(mean.index * 100, mean["macro_f1"], yerr=std["macro_f1"], marker="o",
                color="#2a78d6", capsize=3, label="refuse least confident")
    ax.errorbar(mean.index * 100, mean["random_macro_f1"], yerr=std["random_macro_f1"], marker="o",
                color="#888780", ls="--", capsize=3, label="refuse at random")
    ax.invert_xaxis()
    ax.set_xlabel("Coverage: % of recordings the model answers")
    ax.set_ylabel("Macro-F1 on answered recordings")
    ax.set_title("CNN: answering only when confident")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "abstention")

    cols = ["macro_f1", "challenge_f1", "random_macro_f1"] + [f"refused_{NAMES[c]}" for c in CLASSES]
    print("\n=== A. Abstention (mean over 3 seeds) ===")
    print(mean[cols].round(3).to_string())


# B. Performance by signal quality
def quality_bins(cnn, hrv, feats):
    q = feats[(feats["label"] != "~") & feats["quality_mean"].notna()].copy()
    labels = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]
    q["quality_bin"] = pd.qcut(q["quality_mean"], 5, labels=labels)

    rows = []
    for model_name, preds in {"CNN": cnn, "HRV GBM": hrv}.items():
        d = preds.merge(q[["record", "quality_bin"]], on="record")
        for (seed, b), g in d.groupby(["seed", "quality_bin"], observed=True):
            rows.append({"model": model_name, "seed": seed, "quality_bin": b, "n": len(g),
                         "macro_f1_rhythms": macro_f1(g["true"], g["pred"], RHYTHMS),
                         "predicted_noisy": (g["pred"] == "~").mean()})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "quality_bins.csv", index=False)

    summary = res.groupby(["model", "quality_bin"], observed=True)[["macro_f1_rhythms", "predicted_noisy"]].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model_name, color in [("CNN", "#2a78d6"), ("HRV GBM", "#5f5e5a")]:
        s = res[res["model"] == model_name].groupby("quality_bin", observed=True)["macro_f1_rhythms"]
        ax.errorbar(range(5), s.mean(), yerr=s.std(), marker="o", capsize=3, color=color, label=model_name)
    ax.set_xticks(range(5), labels)
    ax.set_xlabel("Signal quality group (neurokit score, recordings not labelled Noisy)")
    ax.set_ylabel("Macro-F1 (Normal, AF, Other)")
    ax.set_title("Performance by signal quality")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "quality_bins")

    print("\n=== B. Performance by signal quality (mean over 3 seeds) ===")
    print(summary.round(3).to_string())


# C. Length shortcut check
def length_check(cnn, meta):
    d = cnn.merge(meta[["record", "seconds"]], on="record")
    d["length"] = np.where(d["seconds"] < SHORT_SECONDS, f"short (<{SHORT_SECONDS} s)",
                           f"normal (>={SHORT_SECONDS} s)")
    d["true_group"] = np.where(d["true"] == "~", "truly Noisy", "not Noisy")
    res = (d.groupby(["true_group", "length"])
             .agg(recordings=("pred", "size"), predicted_noisy=("pred", lambda p: (p == "~").mean())))
    res["recordings"] = (res["recordings"] / d["seed"].nunique()).round().astype(int)  # per seed
    res.to_csv(RES / "length_check.csv")
    print("\n=== C. Share predicted Noisy, by true label and length (CNN) ===")
    print(res.round(3).to_string())


def main():
    cnn = pd.read_csv(RES / "cnn_predictions.csv")
    cnn["confidence"] = cnn[[f"p_{c}" for c in CLASSES]].max(axis=1)
    hrv = pd.read_csv(RES / "baseline_predictions.csv").query("model == 'hrv_gbm'")
    feats = pd.read_csv(PROC / "hrv_features.csv")
    meta = pd.read_csv(PROC / "meta.csv")

    pd.set_option("display.width", 200)
    abstention(cnn)
    quality_bins(cnn, hrv, feats)
    length_check(cnn, meta)


if __name__ == "__main__":
    main()
