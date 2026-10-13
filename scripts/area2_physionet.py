"""Area 2, test 1: motion cleaning against a REAL answer key (PhysioNet motion-artifact dataset).

Run:  .venv/bin/python scripts/area2_physionet.py      (~3 min)

Each of 23 recordings has two forehead sensors side by side. During four 60-s periods one sensor is
deliberately moved; the other is left alone (= the answer key). Each sensor has its own accelerometer.
(Sweeney et al. 2012, IEEE TITB.)

Questions:
  DETECTION - does our motion measure (share of the signal predictable from the accelerometer)
              find the shaking periods?  Scored with ROC AUC per recording (0.5 = guessing, 1 = perfect).
  CLEANING  - after removing what the accelerometer predicts, is the moved sensor closer to the
              untouched one?  Scored as correlation with the answer key and as dB improvement in
              signal-to-noise (Sweeney's delta-SNR).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score

from eegqc import gait, quality as Q

DATA = ROOT / "data" / "physionet_motion_artifact"
OUT_R, OUT_F = ROOT / "reports" / "results" / "area2", ROOT / "reports" / "figures"
OUT_R.mkdir(parents=True, exist_ok=True)
SF = 256  # work at 256 Hz (EEG recorded at 2048, accelerometers at 200)
BAND = (0.5, 30)


def load(trial: int):
    d = pd.read_csv(DATA / f"Trial{trial}.csv", header=None).values
    eeg, eeg_trig = d[:, 1:3].T, d[:, 3]
    acc_all = d[:, 4:11]
    n_acc = np.where(np.any(acc_all[:, :6] != 0, axis=1))[0][-1] + 1          # columns are zero-padded
    acc, acc_trig = acc_all[:n_acc, :6].T, acc_all[:n_acc, 6]
    # Align on the experiment start (first rise of each stream's trigger), keep the 540-s experiment.
    e0 = np.where(np.diff(eeg_trig) > 0)[0][0]
    rises, falls = np.where(np.diff(acc_trig) > 0.3)[0], np.where(np.diff(acc_trig) < -0.3)[0]
    a0 = rises[0]
    falls = falls[falls > a0]  # some recordings have a trigger drop before the experiment starts
    end_s = min((len(eeg_trig) - e0) / 2048, (n_acc - a0) / 200, (falls[-1] - a0) / 200)
    eeg = resample_poly(eeg[:, e0:e0 + int(end_s * 2048)], 1, 8, axis=1)       # 2048 -> 256 Hz
    acc = resample_poly(acc[:, a0:a0 + int(end_s * 200)], 32, 25, axis=1)      # 200 -> 256 Hz
    n = min(eeg.shape[1], acc.shape[1])
    # Shaking periods = trigger LOW between the first rise and the last fall.
    t = np.arange(n) / SF
    moving = np.zeros(n, dtype=bool)
    for f, r in zip(falls, list(rises[1:]) + [len(acc_trig)]):
        moving[(t >= (f - a0) / 200) & (t < (r - a0) / 200)] = True
    moving &= t < (falls[-1] - a0) / 200
    return eeg[:, :n], acc[:, :n], moving


from scipy.stats import spearmanr


def acc_activity(acc_3: np.ndarray, win: int) -> np.ndarray:
    """Accelerometer activity per sample: smoothed size of the >1 Hz acceleration (gravity removed)."""
    hp = Q.bandpass(acc_3, SF, 1, 30)
    energy = np.sqrt((hp ** 2).sum(0))
    return np.convolve(energy, np.ones(win) / win, mode="same")


def gated_detrend(x: np.ndarray, active: np.ndarray, cutoff_hz: float = 1.0, margin_s: float = 1.0) -> np.ndarray:
    """Inside accelerometer-detected bursts (+-1 s), remove the slow (<1 Hz) part of the signal.

    The artifact here is a slow level shift caused by pushing the sensor; outside bursts nothing is touched.
    Edges are blended over 0.5 s so the switch itself doesn't create a jump.
    """
    k = int(margin_s * SF)
    mask = np.convolve(active.astype(float), np.ones(2 * k + 1), mode="same") > 0
    ramp = np.convolve(mask.astype(float), np.hanning(SF // 2) / np.hanning(SF // 2).sum(), mode="same")
    slow = Q.bandpass(x[None], SF, 0.05, cutoff_hz)[0]
    return x - ramp * slow


rows, examples = [], {}
for trial in range(1, 24):
    eeg, acc, moving = load(trial)
    eeg_f = Q.bandpass(eeg, SF, *BAND)
    acc_f = Q.bandpass(acc, SF, *BAND)
    busy = [acc_f[3 * k:3 * k + 3, moving].var() / acc_f[3 * k:3 * k + 3, ~moving].var() for k in (0, 1)]
    m = int(np.argmax(busy))
    c = 1 - m
    moved, truth = eeg_f[m], eeg_f[c]
    a, b = np.polyfit(truth[~moving], moved[~moving], 1)
    truth_s = a * truth + b
    artifact = moved - truth_s                                   # the real artifact (answer key)

    # Accelerometer bursts: activity well above its typical (rest) level.
    act = acc_activity(acc[3 * m:3 * m + 3], SF // 4)
    rest_level = np.median(act[~moving])
    rest_spread = 1.4826 * np.median(np.abs(act[~moving] - rest_level))
    active = act > rest_level + 10 * rest_spread

    cleaned = {"none": moved,
               "regression on accelerometer": gait.regress_out_motion(moved[None], acc_f[3 * m:3 * m + 3], SF,
                                                                     max_lag_s=0.1, lag_step_s=0.02)[0],
               "burst-gated slow-drift removal": gated_detrend(moved, active)}

    # Score cleaning only where the real artifact is (windows whose true artifact is in the top quarter).
    n = 2 * SF
    w = len(moved) // n
    art_w = artifact[: w * n].reshape(w, n).std(1)
    big = art_w >= np.quantile(art_w, 0.75)
    big_s = np.repeat(big, n)
    rec = dict(trial=trial, moved_sensor=m + 1, burst_share=active.mean())
    for name, x in cleaned.items():
        err_before = (moved - truth_s)[: w * n][big_s]
        err_after = (x - truth_s)[: w * n][big_s]
        rec[f"corr|{name}"] = np.corrcoef(x[: w * n][big_s], truth_s[: w * n][big_s])[0, 1]
        rec[f"dSNR_dB|{name}"] = 10 * np.log10(err_before.var() / err_after.var())
    # Detection: which per-window measure follows the real artifact size?
    act_w = act[: w * n].reshape(w, n).mean(1)
    size_w = moved[: w * n].reshape(w, n).std(1)
    pred = moved - cleaned["regression on accelerometer"]
    yw, pw = moved[: w * n].reshape(w, n), pred[: w * n].reshape(w, n)
    share_w = np.clip(1 - ((yw - pw) ** 2).sum(1) / ((yw - yw.mean(1, keepdims=True)) ** 2).sum(1), 0, 1)
    for name, v in [("accelerometer activity", act_w), ("signal size", size_w), ("motion share (regression)", share_w)]:
        rec[f"detect_rho|{name}"] = spearmanr(v, art_w)[0]
        rec[f"detect_auc|{name}"] = roc_auc_score(big, v)
    rows.append(rec)
    if trial in (1, 12):
        examples[trial] = (moved, truth_s, cleaned["burst-gated slow-drift removal"], moving)
    print(trial, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_R / "physionet_results.csv", index=False)
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
print("\nMEDIANS over 23 recordings:")
print(df.drop(columns=["trial", "moved_sensor"]).median().round(3).to_string())
for name in ["regression on accelerometer", "burst-gated slow-drift removal"]:
    print(f"{name}: better than nothing (dSNR > 0) in {int((df[f'dSNR_dB|{name}'] > 0).sum())} of 23")

# Figure: one example shaking period, before vs after cleaning, with the answer key
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=False)
for ax, trial in zip(axes, examples):
    moved, truth_s, cleaned, moving = examples[trial]
    start = np.where(moving)[0][0]
    sl = slice(start, start + 30 * SF)
    t = np.arange(sl.stop - sl.start) / SF
    ax.plot(t, moved[sl], color=SERIES[1], lw=1, label="moved sensor (raw)")
    ax.plot(t, cleaned[sl], color=SERIES[0], lw=1.2, label="moved sensor after cleaning")
    ax.plot(t, truth_s[sl], color=INK2, lw=1, ls="--", label="untouched sensor (answer key)")
    r = df[df.trial == trial].iloc[0]
    ax.set_title(f"Trial {trial}: burst-gated slow-drift removal, improvement "
                 f"{r['dSNR_dB|burst-gated slow-drift removal']:.1f} dB where the artifact is", loc="left", fontsize=9)
    ax.set_ylabel("µV")
axes[0].legend(frameon=False, ncol=3, fontsize=8)
axes[1].set_xlabel("seconds into a shaking period")
fig.tight_layout(); fig.savefig(OUT_F / "fig11_physionet_cleaning.png", dpi=130); plt.close(fig)
