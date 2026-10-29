"""
1D CNN on raw ECG windows, evaluated on the same recording-level splits as the baselines.

Pipeline for each split:
    - each recording is cut into non-overlapping 7 s windows (2,100 samples at 300 Hz)
    - each window is normalized on its own (mean 0, standard deviation 1)
    - every window gets the label of its recording
    - 15% of the TRAINING recordings are held out as validation (early stopping)
    - the CNN classifies windows; a recording's prediction is the AVERAGE of its window
      probabilities, so the model never sees the recording length
    - all windows of a recording stay in the same set (train, validation or test)

Three training procedures (same data, same network, only training changes):
    v1  max 40 epochs, patience 6, constant learning rate, early stopping on validation loss.
        Training often stopped after 2-5 useful epochs: the class-weighted validation loss
        on a small validation set is noisy.
    v2  max 60 epochs, patience 15, otherwise as v1. Learning curves showed overfitting after
        ~10 epochs, validation loss rising while validation macro-F1 stayed on a plateau
        (overconfidence), and predictions jumping strongly between epochs.
    v3  cosine learning-rate decay (predictions settle late in training), max 40 epochs,
        patience 10, early stopping on validation macro-F1 averaged over the last 3 epochs.
        Final version.

Input : data/processed/signals.pkl, meta.csv, results/splits.csv
Output: results/cnn_<version>_metrics.csv, _summary.csv,
        results/cnn_<version>_predictions.csv (class probabilities per test recording)
        results/cnn_<version>_history.csv (learning curves for every split)

Run from the repository root:
    python scripts/05_cnn.py --version v3           # final version (default)
    python scripts/05_cnn.py --version v2           # earlier versions, for the record
    python scripts/05_cnn.py --version v3 --quick   # one split, few epochs, to test the code
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

FS, WINDOW_SECONDS = 300, 7.0
WIN = int(FS * WINDOW_SECONDS)                      # 2,100 samples
CLASSES = ["N", "A", "O", "~"]
NAMES = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}
SEEDS, N_FOLDS = [0, 1, 2], 5
BATCH, LR, WEIGHT_DECAY = 128, 1e-3, 1e-3

VERSIONS = {
    "v1": {"max_epochs": 40, "patience": 6, "cosine_lr": False, "stop_on": "val_loss"},
    "v2": {"max_epochs": 60, "patience": 15, "cosine_lr": False, "stop_on": "val_loss"},
    "v3": {"max_epochs": 40, "patience": 10, "cosine_lr": True, "stop_on": "val_f1_smoothed"},
}
SMOOTH = 3                                          # epochs averaged for v3's stopping score

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_windows(signals, rec_idx):
    """Cut the given recordings into normalized windows.
    Returns windows (n, 1, WIN) and, for each window, the index of its recording."""
    wins, owner = [], []
    for i in rec_idx:
        sig = signals[i]
        for k in range(len(sig) // WIN):
            w = sig[k * WIN:(k + 1) * WIN]
            wins.append((w - w.mean()) / (w.std() + 1e-6))   # per-window normalization
            owner.append(i)
    return np.stack(wins)[:, None, :].astype("float32"), np.array(owner)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def block(c_in, c_out):
    """Convolution (15 samples = 50 ms) + batch normalization + ReLU + halve the length."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=15, padding=7),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class ECGNet(nn.Module):
    """6 convolution blocks (receptive field about 3 s), global average pooling, linear classifier."""

    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 32),
            block(32, 64), block(64, 64), block(64, 128),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        h = self.features(x)          # (batch, 128, length / 64)
        h = h.mean(dim=2)             # global average pooling: position no longer matters
        return self.classifier(h)     # raw scores (logits)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def predict_proba(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.softmax(model(X[i:i + 1024]), dim=1))
    return torch.cat(out).cpu().numpy()


def recording_f1(model, X, owner, y_rec):
    """Macro-F1 at recording level (mean of window probabilities per recording)."""
    proba = pd.DataFrame(predict_proba(model, X))
    proba["rec"] = owner
    rec_proba = proba.groupby("rec").mean()
    return f1_score(y_rec[rec_proba.index], rec_proba.values.argmax(axis=1),
                    labels=range(4), average="macro", zero_division=0)


def train(Xtr, ytr, Xva, yva, ova, y_rec, seed, cfg):
    torch.manual_seed(seed)
    model = ECGNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])
                 if cfg["cosine_lr"] else None)

    # balanced class weights, computed on the training windows
    counts = torch.bincount(ytr, minlength=4).float()
    weights = counts.sum() / (4 * counts.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    best_score, best_state, wait, history = -np.inf, None, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        order = torch.randperm(len(Xtr), generator=gen)
        train_losses = []
        for i in range(0, len(order), BATCH):            # mini-batches
            idx = order[i:i + BATCH].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([loss_fn(model(Xva[i:i + 1024]), yva[i:i + 1024]).item()
                                for i in range(0, len(Xva), 1024)])
        history.append({"epoch": epoch, "train_loss": np.mean(train_losses), "val_loss": val_loss,
                        "val_macro_f1": recording_f1(model, Xva, ova, y_rec),
                        "lr": optimizer.param_groups[0]["lr"]})

        # early stopping score: higher is better
        if cfg["stop_on"] == "val_loss":
            score = -val_loss
        else:
            score = np.mean([h["val_macro_f1"] for h in history[-SMOOTH:]])

        if score > best_score:
            best_score, wait = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, epoch + 1, history


def scores(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=range(4), average=None, zero_division=0)
    out = {"macro_f1": f1.mean(), "challenge_f1": f1[:3].mean()}
    out.update({f"f1_{NAMES[c]}": v for c, v in zip(CLASSES, f1)})
    return out


def tensor(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=list(VERSIONS), default="v3")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = dict(VERSIONS[args.version])
    if args.quick:
        cfg["max_epochs"] = 3
    runs = [(0, 0)] if args.quick else [(s, f) for s in SEEDS for f in range(N_FOLDS)]

    print(f"device: {device}   version: {args.version}   settings: {cfg}")
    signals = pd.read_pickle(PROC / "signals.pkl")
    meta = pd.read_csv(PROC / "meta.csv")
    splits = pd.read_csv(RES / "splits.csv")
    assert (meta["record"].values == splits["record"].values).all()
    y_rec = meta["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    metric_rows, pred_rows, hist_rows = [], [], []
    for seed, fold in runs:
        t0 = time.time()
        test_rec = np.where(splits[f"fold_seed{seed}"].values == fold)[0]
        trval_rec = np.where(splits[f"fold_seed{seed}"].values != fold)[0]
        tr_rec, va_rec = train_test_split(trval_rec, test_size=0.15,
                                          stratify=y_rec[trval_rec], random_state=seed)
        assert not set(tr_rec) & set(test_rec) and not set(va_rec) & set(test_rec)

        Xtr, otr = make_windows(signals, tr_rec)
        Xva, ova = make_windows(signals, va_rec)
        Xte, ote = make_windows(signals, test_rec)

        model, n_epochs, hist = train(tensor(Xtr), tensor(y_rec[otr], torch.long),
                                      tensor(Xva), tensor(y_rec[ova], torch.long),
                                      ova, y_rec, seed, cfg)
        hist_rows += [{"seed": seed, "fold": fold, **h} for h in hist]

        # recording-level prediction = mean of its window probabilities
        proba = pd.DataFrame(predict_proba(model, tensor(Xte)), columns=CLASSES)
        proba["rec"] = ote
        rec_proba = proba.groupby("rec")[CLASSES].mean()
        pred = rec_proba.values.argmax(axis=1)
        true = y_rec[rec_proba.index]

        metric_rows.append({"model": f"cnn_{args.version}", "seed": seed, "fold": fold,
                            "epochs": n_epochs, **scores(true, pred)})
        for rec, row, p in zip(rec_proba.index, rec_proba.values, pred):
            pred_rows.append({"model": f"cnn_{args.version}", "seed": seed, "fold": fold,
                              "record": meta.loc[rec, "record"], "true": CLASSES[y_rec[rec]],
                              "pred": CLASSES[p], **{f"p_{c}": v for c, v in zip(CLASSES, row)}})
        m = metric_rows[-1]
        print(f"seed {seed} fold {fold}: macro-F1 {m['macro_f1']:.3f}, F1 AF {m['f1_AF']:.3f}, "
              f"F1 Noisy {m['f1_Noisy']:.3f}, {n_epochs} epochs, {time.time() - t0:.0f} s")

    metrics = pd.DataFrame(metric_rows)
    cols = ["macro_f1", "challenge_f1"] + [f"f1_{NAMES[c]}" for c in CLASSES]
    name = f"cnn_{args.version}" + ("_quick" if args.quick else "")
    metrics.to_csv(RES / f"{name}_metrics.csv", index=False)
    metrics.groupby("model")[cols].agg(["mean", "std"]).round(3).to_csv(RES / f"{name}_summary.csv")
    pd.DataFrame(pred_rows).to_csv(RES / f"{name}_predictions.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(RES / f"{name}_history.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Mean over splits ===")
    print(metrics[cols].mean().round(3).to_string())
    print("\n=== Standard deviation over splits ===")
    print(metrics[cols].std().round(3).to_string())


if __name__ == "__main__":
    main()
