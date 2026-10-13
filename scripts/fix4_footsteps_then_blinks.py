"""Fix 4: remove footsteps first (gait-template subtraction), then detect blinks.

Run:  .venv/bin/python scripts/fix4_footsteps_then_blinks.py      (~10 min)

For each person and speed, three versions of the signal:
  raw        : median reference only
  template   : + footstep template subtraction (head motion sensor only)
  template+R : + regression of what's left against the head motion signals

Checks (no labels needed for the first two):
  1. Step-phase histogram of detected blinks: real blinks spread evenly over the step cycle,
     footstep mistakes pile up at one phase. Measured with the Rayleigh R (0 = even, 1 = all at one phase).
  2. Blink rate while moving vs standing (should be similar).
  3. Eye-sensor answer key (also footstep-cleaned) - how many real blinks found / detections real.
  4. Safety: on STANDING recordings, apply the cleaning with fake step times; the brain response
     and decoding should barely change.
  5. Recovery: while moving, does the brain response / decoding recover part of the drop?
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eegqc import gait, lee, quality as Q, usability as U, validate as V

mne.set_log_level("error")
OUT_R, OUT_F = ROOT / "reports" / "results" / "v4", ROOT / "reports" / "figures"
SUBS = [f"sub-{k:02d}" for k in range(1, 11)]
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
SF = 500
FAKE_STEP_S = 0.36  # a jogging step, used only for the standing safety check


def rayleigh(phases: np.ndarray):
    """R: 0 = evenly spread over the step cycle, 1 = all at the same phase. p: chance of R this big if even."""
    phases = phases[~np.isnan(phases)]
    if len(phases) < 5:
        return np.nan, np.nan
    R = np.abs(np.mean(np.exp(2j * np.pi * phases)))
    return float(R), float(np.exp(-len(phases) * R ** 2))


def eye_key(signal: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    band = Q.bandpass(signal[None], SF, 1, 10)[0]
    spread = 1.4826 * np.median(np.abs(band - np.median(band)))
    from scipy.signal import find_peaks
    return find_peaks(np.abs(band), height=threshold * spread, distance=int(0.5 * SF))[0]


rows, phase_rows = [], []
for sub in SUBS:
    for ses in SESSIONS:
        raw = lee.load_eeg(sub, ses)
        scalp = lee.scalp_channels(raw)
        imu = lee.load_imu_aligned(raw, sub, ses)
        motion = imu.get_data(picks=V.MOTION_CHANNELS)
        ref = Q.median_reference(raw.get_data(picks=scalp))
        veog = (raw.get_data(picks=["VEOGU"]) - raw.get_data(picks=["VEOGL"]))
        strikes = gait.heel_strikes(imu.get_data(picks=gait.VERTICAL)[0], SF)
        standing = len(strikes) == 0
        if standing:  # safety check: pretend there are steps, the cleaning must not hurt
            strikes = np.arange(int(5 * SF), ref.shape[1] - int(5 * SF), int(FAKE_STEP_S * SF))
        phase = gait.step_phase(ref.shape[1], strikes)
        events = lee.erp_events(raw)
        minutes = raw.times[-1] / 60

        cleaned = gait.remove_footsteps(ref, strikes)
        versions = {"raw": ref, "template": cleaned,
                    "template+R": gait.regress_out_motion(cleaned, Q.bandpass(motion, SF, 0.5, 20), SF)}
        key_raw = eye_key(veog[0])
        key_clean = eye_key(gait.remove_footsteps(veog, strikes)[0])

        pos = np.array([raw.info["chs"][raw.ch_names.index(c)]["loc"][:3] for c in scalp])
        for vname, data in versions.items():
            blinks = U.detect_blinks(data, scalp, SF)
            R, p = rayleigh(phase[blinks]) if not standing else (np.nan, np.nan)
            recall, precision = V.match_events(blinks, key_clean, SF)
            trials, y, onsets, times = V.erp_trials_from_array(data, SF, events, scalp)
            auc = V.auc(y, V.out_of_fold_proba(V.decoding_features(trials, times), y))
            rec = U.Recording(data, SF, scalp, pos, motion=motion)
            st, n = rec.windows()
            use, _, _ = rec.measure(st, n)
            rows.append(dict(sub=sub, ses=ses, speed=lee.SPEEDS[ses], standing=standing, version=vname,
                             blinks_per_min=len(blinks) / minutes, phase_R=R, phase_p=p,
                             key_per_min=len(key_clean) / minutes, key_raw_per_min=len(key_raw) / minutes,
                             found_pct=100 * recall, real_pct=100 * precision,
                             p300_d_Pz=V.p300_effect(trials[:, scalp.index("Pz")], y, times), auc=auc,
                             motion_r2=float(np.median(use["motion_r2"])),
                             unusable_pct=100 * (U.usability_score(use) < 0.5).mean()))
            if not standing:
                phase_rows += [dict(sub=sub, ses=ses, version=vname, phase=ph) for ph in phase[blinks] if not np.isnan(ph)]
        print(sub, ses, "steps" if not standing else "standing (fake steps)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_R / "fix4_cleaning.csv", index=False)
ph = pd.DataFrame(phase_rows)
ph.to_csv(OUT_R / "fix4_blink_step_phases.csv", index=False)

pd.set_option("display.width", 230)
summary = df.groupby(["speed", "version"], sort=False)[
    ["blinks_per_min", "phase_R", "key_per_min", "found_pct", "real_pct", "p300_d_Pz", "auc", "motion_r2", "unusable_pct"]].median()
summary.to_csv(OUT_R / "fix4_summary.csv")
print(summary.round(3).to_string())

# blink rate while moving vs standing, per person
stand = df[df.standing].set_index(["sub", "version"]).blinks_per_min
jog = df[df.ses == "ses-05"].set_index(["sub", "version"]).blinks_per_min
ratio = (jog / stand).unstack()
ratio.to_csv(OUT_R / "fix4_blink_rate_jog_over_stand.csv")
print("\nblink rate jogging / standing, per person:\n", ratio.round(2).to_string())

# Figure: step-phase histograms of detected 'blinks' while jogging, before vs after cleaning
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
jog_ph = ph[ph.ses == "ses-05"]
for ax, (vname, color) in zip(axes, zip(["raw", "template", "template+R"], SERIES)):
    v = jog_ph[jog_ph.version == vname].phase.values
    ax.hist(v * 100, bins=20, range=(0, 100), color=color, edgecolor=SURFACE, linewidth=1.5)
    R, _ = rayleigh(v)
    ax.axhline(len(v) / 20, color=INK2, lw=1, ls="--")
    ax.set_title(f"{vname}: {len(v)} 'blinks', R = {R:.2f}", loc="left", fontsize=9)
    ax.set_xlabel("where in the step cycle (% from heel strike)")
axes[0].set_ylabel("detected blinks (all 10 people, jogging)")
fig.suptitle("Real blinks spread evenly (dashed line); footstep mistakes pile up at one phase", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT_F / "fig8_blink_step_phase.png", dpi=130); plt.close(fig)

# Figure: what cleaning does to the brain response and decoding, by speed
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
speeds = [lee.SPEEDS[s] for s in SESSIONS]
for ax, col, label in [(axes[0], "p300_d_Pz", "brain-response strength at Pz (d)"), (axes[1], "auc", "decoding AUC")]:
    for vname, color in zip(["raw", "template", "template+R"], SERIES):
        m = df[df.version == vname].groupby("speed")[col].mean().loc[speeds]
        ax.plot(range(4), m.values, marker="o", color=color, lw=2, label=vname)
    ax.set_xticks(range(4), ["stand*\n0", "slow walk\n0.8", "fast walk\n1.6", "jog\n2.0"]); ax.set_ylabel(label)
axes[0].legend(frameon=False)
fig.suptitle("Footstep cleaning (mean of 10 people). *standing = safety check with fake step times", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(OUT_F / "fig9_cleaning_effect.png", dpi=130); plt.close(fig)
