import time
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FS = 300
MIN_BEATS = 5                    
RR_PLAUSIBLE = (0.25, 2.0)       


MODEL_FEATURES = ["mean_rr", "sdnn", "rmssd", "pnn50", "cv_rr", "median_abs_diff_rr",
                  "frac_implausible_rr", "quality_mean"]


def rr_features(rpeaks):
    """Rhythm features from R peak positions (in samples)."""
    rr = np.diff(rpeaks) / FS                    
    drr = np.diff(rr)                             
    plausible = (rr >= RR_PLAUSIBLE[0]) & (rr <= RR_PLAUSIBLE[1])
    return {
        "n_beats": len(rpeaks),
        "mean_rr": rr.mean(),
        "sdnn": rr.std(),                                         
        "rmssd": np.sqrt(np.mean(drr ** 2)),                      
        "pnn50": np.mean(np.abs(drr) > 0.05),                  
        "cv_rr": rr.std() / rr.mean(),                            
        "median_abs_diff_rr": np.median(np.abs(drr)),             
        "frac_implausible_rr": 1 - plausible.mean(),              
    }


def process(signal):
    """Return (features dict, status string) for one recording."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(signal, sampling_rate=FS)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=FS)
            rpeaks = np.asarray(info["ECG_R_Peaks"])
            if len(rpeaks) < MIN_BEATS:
                return {"n_beats": len(rpeaks)}, "too_few_beats"
            feats = rr_features(rpeaks)
            quality = nk.ecg_quality(cleaned, rpeaks=rpeaks, sampling_rate=FS)
            feats["quality_mean"] = float(np.nanmean(quality))
            return feats, "ok"
    except Exception as e:                       # keep going, but record what happened
        return {}, f"error: {type(e).__name__}"


def main():
    signals = pd.read_pickle(PROC / "signals.pkl")
    meta = pd.read_csv(PROC / "meta.csv")
    assert len(signals) == len(meta)

    rows, t0 = [], time.time()
    for i, (record, sig) in enumerate(zip(meta["record"], signals)):
        feats, status = process(sig)
        rows.append({"record": record, "status": status, **feats})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(signals)} recordings, {time.time() - t0:.0f} s")

    feats = pd.DataFrame(rows)
    out = meta[["record", "label", "class_name", "seconds"]].merge(feats, on="record")
    out.to_csv(PROC / "hrv_features.csv", index=False)

    print("\n=== Status per class ===")
    print(pd.crosstab(out["class_name"], out["status"]))
    print("\n=== Median feature values per class (successful recordings) ===")
    ok = out[out["status"] == "ok"]
    print(ok.groupby("class_name")[MODEL_FEATURES].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
