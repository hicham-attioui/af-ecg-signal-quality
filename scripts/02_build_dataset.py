import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import StratifiedKFold

# Paths and settings
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "training"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

FS = 300                   
TRIM_SECONDS = 1.0         
WINDOW_SECONDS = 7.0        
LABELS = {"N": "Normal", "A": "AF", "O": "Other rhythm", "~": "Noisy"}
SEEDS, N_FOLDS = [0, 1, 2], 5


# Steps

def load_labels():
    """REFERENCE-v3.csv, with record names normalised to 'A00001' (some versions write 'A00/A00001')."""
    ref = pd.read_csv(RAW / "REFERENCE-v3.csv", header=None, names=["record", "label"])
    ref["record"] = ref["record"].str.split("/").str[-1]
    assert ref["record"].is_unique
    assert set(ref["label"]) <= set(LABELS), f"Unexpected labels: {set(ref['label']) - set(LABELS)}"
    ref["class_name"] = ref["label"].map(LABELS)
    return ref


def load_signal(record):
    """Read one recording: a single lead at 300 Hz, returned as float32."""
    rec = wfdb.rdrecord(str(RAW / record[:3] / record))
    assert rec.fs == FS, f"{record}: unexpected sampling rate {rec.fs}"
    assert rec.n_sig == 1, f"{record}: expected one lead, found {rec.n_sig}"
    return rec.p_signal[:, 0].astype("float32")


def trim(signal):
    n = int(TRIM_SECONDS * FS)
    return signal[n:-n]


def make_splits(labels):
    """Stratified 5-fold CV at RECORDING level, repeated with 3 seeds."""
    splits = pd.DataFrame(index=labels.index)
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        fold = np.empty(len(labels), dtype=int)
        for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            fold[test_idx] = f
        splits[f"fold_seed{seed}"] = fold
    return splits



# Main

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)

    meta = load_labels()
    print(f"Loading {len(meta)} recordings...")

    signals, raw_len, trimmed_len, hashes = [], [], [], []
    for record in meta["record"]:
        sig = load_signal(record)
        hashes.append(hashlib.md5(sig.tobytes()).hexdigest())   
        raw_len.append(len(sig))
        sig = trim(sig)
        trimmed_len.append(len(sig))
        signals.append(sig)

    meta["raw_seconds"] = np.array(raw_len) / FS
    meta["seconds"] = np.array(trimmed_len) / FS
    meta["md5"] = hashes

    
    dup = meta[meta.duplicated("md5", keep=False)].sort_values(["md5", "record"])
    keep = ~meta.duplicated("md5", keep="first").values
    meta = meta[keep].reset_index(drop=True)
    signals = [s for s, k in zip(signals, keep) if k]
    assert len(signals) == len(meta)

    meta["n_windows"] = (meta["seconds"] // WINDOW_SECONDS).astype(int)   
    assert (meta["n_windows"] >= 1).all(), "Some recordings are shorter than one window"

    splits = make_splits(meta["label"])
    splits.insert(0, "record", meta["record"])

    pd.to_pickle(signals, PROC / "signals.pkl")
    meta.to_csv(PROC / "meta.csv", index=False)
    splits.to_csv(RES / "splits.csv", index=False)
    dup.to_csv(RES / "duplicates.csv", index=False)

    # Summary
    print("\n=== Summary ===")
    print(f"Exact duplicates: {len(dup)} recordings in {dup['md5'].nunique()} groups, "
          f"{len(dup) - dup['md5'].nunique()} removed")
    if len(dup):
        print(dup[["record", "class_name", "raw_seconds", "md5"]].to_string(index=False))
    print(f"\nRecordings kept: {len(meta)}   (sampling rate {FS} Hz, one lead)")
    print("\nClass counts:")
    print(meta["class_name"].value_counts().to_string())
    print(f"\nLength after trimming {TRIM_SECONDS:.1f} s at each end (seconds):")
    print(meta["seconds"].describe().round(2).to_string())
    print(f"\n{WINDOW_SECONDS:.1f} s windows per recording:")
    print(meta["n_windows"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
