"""Fix 3: can the score rank trials WITHIN a recording?

Run:  .venv/bin/python scripts/fix3_ranking.py      (~10 min)

Changes tested (one at a time, so we can see which matters):
  resolution : pass/fail score (mostly 0 or 1)  vs  severity (how far past the yellow line, no ties)
  alignment  : fixed 2-s grid                   vs  each trial's exact span (-0.2..0.8 s)
Targets (continuous):
  p_correct  : decoder's out-of-fold probability for the correct class (decoder = judge only)
  loo_dist   : distance from the person's average response, trial left out (no model at all)
Baseline     : peak-to-peak size of each trial. If even that can't rank trials, steady treadmill walking
               simply doesn't create trial-to-trial differences.
Direct test  : inject fake artifacts of known size into clean standing trials; does severity order them?
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
from scipy.stats import binomtest, spearmanr

from eegqc import lee, usability as U, validate as V

mne.set_log_level("error")
OUT_R, OUT_F = ROOT / "reports" / "results" / "v4", ROOT / "reports" / "figures"
OUT_R.mkdir(parents=True, exist_ok=True)
SUBS = [f"sub-{k:02d}" for k in range(1, 11)]
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
SF = 500
rng = np.random.default_rng(0)


def build(data, raw, sub, ses, scalp, with_motion=True):
    pos = np.array([raw.info["chs"][raw.ch_names.index(c)]["loc"][:3] for c in scalp])
    motion = lee.load_imu_aligned(raw, sub, ses).get_data(picks=V.MOTION_CHANNELS) if with_motion else None
    return U.Recording(data, SF, scalp, pos, ceiling=lee.recorder_limits(sub, ses, channels=scalp), motion=motion)


# ============================================================ part A: ranking real trials
rows = []
for sub in SUBS:
    for ses in SESSIONS:
        raw = lee.load_eeg(sub, ses)
        scalp = lee.scalp_channels(raw)
        rec = build(raw.get_data(picks=scalp), raw, sub, ses, scalp)
        trials, y, onsets, times = V.erp_trials(raw, scalp)
        proba = V.out_of_fold_proba(V.decoding_features(trials, times), y)
        target = {"p_correct": V.p_correct(proba, y),
                  "loo_dist": V.loo_distance(trials, y, times, [scalp.index(c) for c in V.P300_SENSORS])}

        # 2-s grid: each trial gets the worst window it overlaps
        wst, wn = rec.windows()
        wuse, _, _ = rec.measure(wst, wn)
        grid_pf = V.trial_scores(U.usability_score(wuse), onsets, SF)            # (trials, sensors), 1 = good
        grid_sev = -V.trial_scores(-U.usability_severity(wuse), onsets, SF)      # worst (= max) severity
        # exact span of each trial
        euse, _, _ = rec.measure(onsets + int(V.TMIN * SF), int((V.TMAX - V.TMIN) * SF))
        exact_pf, exact_sev = U.usability_score(euse).T, U.usability_severity(euse).T

        measures = {  # all oriented so that LARGER = WORSE trial
            "pass/fail, 2-s grid": 1 - grid_pf.mean(1),
            "pass/fail, exact span": 1 - exact_pf.mean(1),
            "severity, 2-s grid": grid_sev.mean(1),
            "severity, exact span": exact_sev.mean(1),
            "motion share only, exact span": euse["motion_r2"].mean(0),
            "baseline: peak-to-peak": np.ptp(trials, axis=-1).mean(1),
        }
        for mname, m in measures.items():
            for tname, t in target.items():
                good_direction = -1 if tname == "p_correct" else 1   # worse trial -> lower p_correct / larger distance
                rho = spearmanr(m, t)[0] if np.ptp(m) > 0 else np.nan
                rows.append(dict(sub=sub, ses=ses, speed=lee.SPEEDS[ses], measure=mname, target=tname,
                                 rho_in_expected_direction=good_direction * rho,
                                 distinct_values=len(np.unique(np.round(m, 6)))))
        print(sub, ses, flush=True)

A = pd.DataFrame(rows)
A.to_csv(OUT_R / "fix3_ranking_per_recording.csv", index=False)
summary = []
for (mname, tname), g in A.groupby(["measure", "target"], sort=False):
    r = g.rho_in_expected_direction.dropna()
    k = int((r > 0).sum())
    summary.append(dict(measure=mname, target=tname, median_rho=r.median(), right_direction=f"{k} of {len(r)}",
                        sign_test_p=binomtest(k, len(r), 0.5).pvalue if len(r) else np.nan,
                        median_distinct_values=g.distinct_values.median()))
S = pd.DataFrame(summary)
S.to_csv(OUT_R / "fix3_ranking_summary.csv", index=False)
pd.set_option("display.width", 220)
print(S.round(3).to_string(index=False))
print("\nby speed (severity, exact span):")
print(A[A.measure == "severity, exact span"].groupby(["target", "speed"]).rho_in_expected_direction.median().round(3).unstack())

# ============================================================ part B: fake artifacts of known size
SIZES_UV = [0, 5, 10, 20, 40, 80, 160]
N_PER_SIZE = 8
t_art = np.arange(int(1.0 * SF)) / SF  # artifacts placed within the trial's 1-s span


def waveform(kind, size_uv):
    a = size_uv * 1e-6
    if kind == "pop":
        w = np.zeros_like(t_art); w[250:255] = a; return w                      # 10 ms spike
    if kind == "slow wave (motion-like)":
        return a * np.sin(2 * np.pi * 2.75 * t_art) * np.hanning(len(t_art))    # jogging step rate
    if kind == "muscle":
        noise = rng.standard_normal(len(t_art))
        f = np.fft.rfftfreq(len(t_art), 1 / SF); spec = np.fft.rfft(noise); spec[(f < 15) | (f > 45)] = 0
        w = np.fft.irfft(spec, len(t_art)); return a * w / w.std()
    if kind == "drift":
        return a * np.sin(np.pi * t_art / t_art[-1])                             # one slow 1-s hump
    if kind == "blink":
        return a * np.exp(-0.5 * ((t_art - 0.5) / 0.06) ** 2)                   # 60 ms-wide bump


TARGET_SENSOR = {"pop": "Cz", "slow wave (motion-like)": "Cz", "muscle": "Cz", "drift": "Cz", "blink": "Fp1"}
BLINK_SPREAD = {"Fp1": 1.0, "Fp2": 1.0, "AFz": 0.4, "F3": 0.2, "Fz": 0.2, "F4": 0.2}

inj = []
for sub in SUBS:
    ses = "ses-02"  # standing: the cleanest recordings
    raw = lee.load_eeg(sub, ses)
    scalp = lee.scalp_channels(raw)
    base = raw.get_data(picks=scalp)
    events = lee.erp_events(raw)
    starts = events[:, 0] + int(V.TMIN * SF)
    n = int((V.TMAX - V.TMIN) * SF)
    rec0 = build(base, raw, sub, ses, scalp, with_motion=False)
    sev0 = U.usability_severity(rec0.measure(starts, n)[0])          # (sensors, trials)
    clean = np.argsort(sev0.sum(0))[: len(SIZES_UV) * N_PER_SIZE]      # cleanest trials
    rng.shuffle(clean)
    for kind, sensor in TARGET_SENSOR.items():
        data = base.copy()
        size_of_trial = {}
        for j, tr in enumerate(clean):
            size = SIZES_UV[j % len(SIZES_UV)]
            size_of_trial[tr] = size
            w = waveform(kind, size)
            s0 = events[tr, 0] - int(0.2 * SF)
            spread = BLINK_SPREAD if kind == "blink" else {sensor: 1.0}
            for ch, frac in spread.items():
                data[scalp.index(ch), s0:s0 + len(w)] += frac * w
        rec = build(data, raw, sub, ses, scalp, with_motion=False)
        sev = U.usability_severity(rec.measure(starts, n)[0])
        i = scalp.index(sensor)
        for tr, size in size_of_trial.items():
            inj.append(dict(sub=sub, kind=kind, size_uv=size, severity=sev[i, tr], severity_before=sev0[i, tr]))
    print("injection", sub, flush=True)

B = pd.DataFrame(inj)
B.to_csv(OUT_R / "fix3_injection.csv", index=False)
table = B.groupby(["kind", "size_uv"]).severity.median().unstack()
order = {k: spearmanr(g.size_uv, g.severity)[0] for k, g in B.groupby("kind")}
print("\nmedian severity at the injected sensor, by artifact size (µV):")
print(table.round(3).to_string())
print("Spearman(size, severity):", {k: round(v, 2) for k, v in order.items()})

# figure
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
fig, ax = plt.subplots(figsize=(7.5, 4))
for color, kind in zip(SERIES, TARGET_SENSOR):
    med = table.loc[kind]
    ax.plot(range(len(SIZES_UV)), med.values, marker="o", color=color, lw=2, label=f"{kind} (rank corr {order[kind]:.2f})")
ax.set_xticks(range(len(SIZES_UV)), SIZES_UV); ax.set_xlabel("size of the injected artifact (µV)")
ax.set_ylabel("usability severity at that sensor\n(0 = nothing past a yellow line)")
ax.set_title("Fake artifacts of known size: does the severity put them in order?", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT_F / "fig7_injection_ranking.png", dpi=130); plt.close(fig)
