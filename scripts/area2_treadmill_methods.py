"""Area 2, test 2: motion cleaning methods compared on the treadmill data (15 people).

Run:  .venv/bin/python scripts/area2_treadmill_methods.py      (~30-40 min)

Methods (all rules / standard algorithms, nothing trained on labels):
  raw            : median reference only
  steps+R always : footstep template subtraction + regression on head motion, every sensor (fix 4)
  steps+R gated  : the same, but only on sensors whose motion share is above the yellow line (0.3)
  ASR            : Artifact Subspace Reconstruction (meegkit, cutoff 20), calibrated on the first
                   2 minutes of the same person's STANDING recording
  ICA (head-IMU) : 20 independent components; remove components whose time course is predictable from
                   the head motion sensor (motion share > 0.3, the same yellow line)
Judged by: decoding AUC and brain-response strength (P300 effect at Pz). Standing = safety check
(footstep methods use fake step times there; the gate should simply leave it untouched).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np
import pandas as pd
from meegkit.asr import ASR
from scipy.stats import wilcoxon

from eegqc import gait, lee, quality as Q, usability as U, validate as V

mne.set_log_level("error")
OUT_R = ROOT / "reports" / "results" / "area2"
OUT_R.mkdir(parents=True, exist_ok=True)
SUBS = [f"sub-{k:02d}" for k in list(range(1, 11)) + list(range(16, 21))]
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
SF = 500
GATE = U.USABILITY_CHECKS["motion_r2"].warn  # 0.3


def motion_share(x: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Per row of x: share of its 0.5-20 Hz signal predictable from head motion (cross-validated)."""
    xb = Q.bandpass(x, SF, 0.5, 20)
    resid = gait.regress_out_motion(xb, Q.bandpass(motion, SF, 0.5, 20), SF)
    return np.clip(1 - resid.var(1) / xb.var(1), 0, 1)


def imu_guided_ica(data: np.ndarray, motion: np.ndarray, scalp: list[str]) -> tuple[np.ndarray, int]:
    info = mne.create_info(scalp, SF, "eeg")
    fit_raw = mne.io.RawArray(Q.bandpass(data, SF, 1, 40), info).resample(125)
    ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=0, max_iter=800)
    ica.fit(fit_raw)
    full = mne.io.RawArray(data, info)
    sources = ica.get_sources(full).get_data()
    bad = list(np.where(motion_share(sources, motion) > GATE)[0])
    ica.exclude = bad
    return ica.apply(full.copy()).get_data(), len(bad)


rows = []
for sub in SUBS:
    calib = None
    for ses in SESSIONS:
        vhdr = lee.ROOT / "sourcedata" / sub / ses / "eeg" / f"{sub}_{ses}_task-ERP_eeg.vhdr"
        if not vhdr.exists():
            continue
        raw = lee.load_eeg(sub, ses)
        scalp = lee.scalp_channels(raw)
        motion = lee.load_imu_aligned(raw, sub, ses).get_data(picks=V.MOTION_CHANNELS)
        ref = Q.median_reference(raw.get_data(picks=scalp))
        events = lee.erp_events(raw)
        if ses == "ses-02":
            calib = Q.bandpass(ref[:, : 120 * SF], SF, 0.5, 40)
        strikes = gait.heel_strikes(lee.load_imu_aligned(raw, sub, ses).get_data(picks=gait.VERTICAL)[0], SF)
        standing = len(strikes) == 0
        if standing:
            strikes = np.arange(5 * SF, ref.shape[1] - 5 * SF, int(0.36 * SF))

        always = gait.regress_out_motion(gait.remove_footsteps(ref, strikes), Q.bandpass(motion, SF, 0.5, 20), SF)
        share = motion_share(ref, motion)
        gate = share > GATE
        gated = np.where(gate[:, None], always, ref)
        asr = ASR(sfreq=SF, cutoff=20, method="euclid")
        asr.fit(calib)
        asr_out = asr.transform(Q.bandpass(ref, SF, 0.5, 40))
        ica_out, n_removed = imu_guided_ica(ref, motion, scalp)

        for name, data in [("raw", ref), ("steps+R always", always), ("steps+R gated", gated),
                           ("ASR", asr_out), ("ICA (head-IMU)", ica_out)]:
            trials, y, onsets, times = V.erp_trials_from_array(data, SF, events, scalp)
            rows.append(dict(sub=sub, ses=ses, speed=lee.SPEEDS[ses], method=name,
                             auc=V.auc(y, V.out_of_fold_proba(V.decoding_features(trials, times), y)),
                             p300_d_Pz=V.p300_effect(trials[:, scalp.index("Pz")], y, times),
                             sensors_gated=int(gate.sum()), ica_removed=n_removed,
                             motion_share_median=float(np.median(share))))
        print(sub, ses, f"gated sensors {gate.sum():2d}, ICA removed {n_removed}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_R / "treadmill_methods.csv", index=False)

pd.set_option("display.width", 220)
print("\nMEAN by speed and method")
print(df.pivot_table(index="speed", columns="method", values="auc", sort=False).round(3).to_string())
print(df.pivot_table(index="speed", columns="method", values="p300_d_Pz", sort=False).round(3).to_string())
print("\nPAIRED vs raw (decoding AUC): mean change, better in k of n, Wilcoxon p")
out = []
for speed in df.speed.unique():
    p = df[df.speed == speed].pivot(index="sub", columns="method", values="auc")
    for m in ["steps+R always", "steps+R gated", "ASR", "ICA (head-IMU)"]:
        d = (p[m] - p["raw"]).dropna()
        pval = wilcoxon(d).pvalue if (d != 0).sum() > 0 else np.nan
        out.append(dict(speed=speed, method=m, mean_change=d.mean(), better=f"{int((d > 0).sum())} of {len(d)}",
                        same=int((d == 0).sum()), p=pval))
paired = pd.DataFrame(out)
paired.to_csv(OUT_R / "treadmill_methods_paired.csv", index=False)
print(paired.round(3).to_string(index=False))
print("\nsensors gated (median by speed):", df[df.method == "raw"].groupby("speed", sort=False).sensors_gated.median().to_dict())
print("ICA components removed (median by speed):", df[df.method == "raw"].groupby("speed", sort=False).ica_removed.median().to_dict())
