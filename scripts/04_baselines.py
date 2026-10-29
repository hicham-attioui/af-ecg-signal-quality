import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

CLASSES = ["N", "A", "O", "~"]                       
NAMES = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}
SEEDS, N_FOLDS = [0, 1, 2], 5

FEATURES = ["mean_rr", "sdnn", "rmssd", "pnn50", "cv_rr", "median_abs_diff_rr",
            "frac_implausible_rr", "quality_mean"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_model(name, seed):
    if name == "hrv_lr":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),   
            ("scale", StandardScaler()),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-4, 2, 8), l1_ratios=(0,), solver="lbfgs",
                cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                scoring="f1_macro", class_weight="balanced", max_iter=5000,
                n_jobs=-1, use_legacy_attributes=True)),
        ])
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)


def scores(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    out = {"macro_f1": f1.mean(), "challenge_f1": f1[:3].mean()}
    out.update({f"f1_{NAMES[c]}": v for c, v in zip(CLASSES, f1)})
    return out


def main():
    feats = pd.read_csv(PROC / "hrv_features.csv")
    splits = pd.read_csv(RES / "splits.csv")
    data = splits.merge(feats, on="record")
    assert len(data) == len(splits) == len(feats), "records do not match between files"
    y = data["label"].values

    inputs = {"hrv_lr": FEATURES, "hrv_gbm": FEATURES, "length_gbm": ["seconds"]}

    metric_rows, pred_rows = [], []
    for model_name, cols in inputs.items():
        X, t0 = data[cols].values, time.time()
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                test = data[f"fold_seed{seed}"].values == fold
                model = make_model(model_name, seed).fit(X[~test], y[~test])
                pred = model.predict(X[test])
                metric_rows.append({"model": model_name, "seed": seed, "fold": fold, **scores(y[test], pred)})
                pred_rows += [{"model": model_name, "seed": seed, "fold": fold, "record": r,
                               "true": t, "pred": p}
                              for r, t, p in zip(data.loc[test, "record"], y[test], pred)]
        print(f"{model_name:11s} done in {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(metric_rows)
    cols = ["macro_f1", "challenge_f1"] + [f"f1_{NAMES[c]}" for c in CLASSES]
    summary = metrics.groupby("model")[cols].agg(["mean", "std"]).round(3)

    metrics.to_csv(RES / "baseline_metrics.csv", index=False)
    summary.to_csv(RES / "baseline_summary.csv")
    pd.DataFrame(pred_rows).to_csv(RES / "baseline_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Mean over 15 splits ===")
    print(metrics.groupby("model")[cols].mean().round(3).to_string())
    print("\n=== Standard deviation over 15 splits ===")
    print(metrics.groupby("model")[cols].std().round(3).to_string())


if __name__ == "__main__":
    main()
