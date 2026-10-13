"""Fixes 1 + 2: usability score tied to the decoder's band, and hum compared across sensors.

Run:  .venv/bin/python scripts/fix12_usability_and_hum.py      (~10 min for 10 people)

Fix 1 check: per person, how closely does the score track decoding accuracy across the 4 speeds?
             old score (version 3, all checks) vs new usability score (0.5-20 Hz).
Fix 2 check: in recordings with high room hum, how many sensors does the hum check now flag
             (old: fixed threshold; new: compared with the other sensors)? Do the flagged sensors
             also fail other checks or get flagged by pyprep?
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np
import pandas as pd
from pyprep.find_noisy_channels import NoisyChannels
from scipy.stats import spearmanr

from eegqc import lee, usability as U, validate as V

mne.set_log_level("error")
OUT = ROOT / "reports" / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)
SUBS = [f"sub-{k:02d}" for k in range(1, 11)]
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
OLD = pd.read_csv(ROOT / "reports" / "results" / "test_B_decoding.csv")
OLD = OLD[OLD.score == "all checks"].set_index(["sub", "ses"])


def old_hum_flag(rec: U.Recording, starts, n) -> np.ndarray:
    """Version-3 rule: hum power / 1-40 Hz power on the median reference, fixed yellow line 0.10."""
    seg = U._segments(rec.ref, starts, n)
    f, spec = U._power(seg - seg.mean(-1, keepdims=True), rec.sfreq)
    lf = rec.line_freq
    ratio = spec[..., (f >= lf - 1) & (f <= lf + 1)].sum(-1) / spec[..., (f >= 1) & (f <= 40)].sum(-1)
    return ratio > 0.10


rows, sensor_rows = [], []
for sub in SUBS:
    for ses in SESSIONS:
        raw = lee.load_eeg(sub, ses)
        scalp = lee.scalp_channels(raw)
        imu = lee.load_imu_aligned(raw, sub, ses)
        pos = np.array([raw.info["chs"][raw.ch_names.index(c)]["loc"][:3] for c in scalp])
        data = raw.get_data(picks=scalp)
        rec = U.Recording(data, raw.info["sfreq"], scalp, pos, ceiling=lee.recorder_limits(sub, ses, channels=scalp),
                          motion=imu.get_data(picks=V.MOTION_CHANNELS))
        starts, n = rec.windows()
        use, hw, room = rec.measure(starts, n)
        score = U.usability_score(use)
        flags = U.hardware_flags(hw)

        # Decoding (the judge) and the score of each trial (worst 2-s window it overlaps, as in version 3)
        trials, y, onsets, times = V.erp_trials(raw, scalp)
        auc = V.auc(y, V.out_of_fold_proba(V.decoding_features(trials, times), y))
        new_trial_q = V.trial_scores(score, onsets, rec.sfreq).mean()

        # Fix 2: hum flags, old vs new
        old_flag = old_hum_flag(rec, starts, n)
        hum_sensors = [scalp[i] for i in np.where(flags["hum_z"].mean(1) > 0.5)[0]]
        nc = NoisyChannels(mne.io.RawArray(data, raw.copy().pick(scalp).info), do_detrend=True, random_state=0)
        nc.find_all_bads(ransac=False)
        pyprep_bad = set(nc.get_bads())
        usability_bad_pct = 100 * (score < 0.5).mean(1)
        for s in hum_sensors:
            i = scalp.index(s)
            sensor_rows.append(dict(sub=sub, ses=ses, sensor=s, pct_windows_hum_flagged=100 * flags["hum_z"][i].mean(),
                                    usability_bad_pct=usability_bad_pct[i],
                                    usability_bad_pct_median_sensor=np.median(usability_bad_pct),
                                    also_fails_other_checks=usability_bad_pct[i] > 2 * np.median(usability_bad_pct) + 5,
                                    pyprep_flags_it=s in pyprep_bad))
        rows.append(dict(sub=sub, ses=ses, speed=lee.SPEEDS[ses], line_freq=rec.line_freq,
                         room_hum=float(np.median(room)), room_warning=bool(np.median(room) > U.ROOM_HUM_WARN),
                         old_hum_flag_pct=100 * old_flag.mean(), new_hum_flag_pct=100 * flags["hum_z"].mean(),
                         new_hum_sensors=" ".join(hum_sensors), pyprep_bad=" ".join(sorted(pyprep_bad)),
                         old_bad_pct=np.nan, new_unusable_pct=100 * (score < 0.5).mean(),
                         old_trial_quality=OLD.loc[(sub, ses), "mean_trial_quality"], new_trial_quality=new_trial_q,
                         auc=auc))
        print(sub, ses, f"room {np.median(room):6.2f}  hum flags old {100 * old_flag.mean():4.1f}% new {100 * flags['hum_z'].mean():4.1f}%  "
              f"unusable {100 * (score < 0.5).mean():4.1f}%  auc {auc:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "fix12_recordings.csv", index=False)
pd.DataFrame(sensor_rows).to_csv(OUT / "fix2_hum_flagged_sensors.csv", index=False)

# Fix 1: per person, Spearman correlation across the 4 speeds (score vs decoding AUC)
per_person = []
for sub, g in df.groupby("sub"):
    per_person.append(dict(sub=sub, rho_old=spearmanr(g.old_trial_quality, g.auc)[0],
                           rho_new=spearmanr(g.new_trial_quality, g.auc)[0]))
pp = pd.DataFrame(per_person)
pp.to_csv(OUT / "fix1_tracking_per_person.csv", index=False)

pd.set_option("display.width", 220)
print("\nFIX 1 - score vs decoding across speeds, per person (Spearman):")
print(pp.round(2).to_string(index=False))
print(f"old: positive in {(pp.rho_old > 0).sum()} of 10, median {pp.rho_old.median():.2f} | "
      f"new: positive in {(pp.rho_new > 0).sum()} of 10, median {pp.rho_new.median():.2f}")
print("\nFIX 2 - hum flags in recordings with a room warning:")
w = df[df.room_warning]
print(w[["sub", "ses", "room_hum", "old_hum_flag_pct", "new_hum_flag_pct", "new_hum_sensors", "pyprep_bad"]].round(2).to_string(index=False))
print(f"recordings with room warning: {len(w)} of {len(df)} | old flags {w.old_hum_flag_pct.mean():.1f}% -> new {w.new_hum_flag_pct.mean():.1f}% of sensor-windows")
s = pd.DataFrame(sensor_rows)
if len(s):
    print(f"sensors flagged for hum (all recordings): {len(s)}; also fail other usability checks: {s.also_fails_other_checks.sum()}; "
          f"also flagged by pyprep: {s.pyprep_flags_it.sum()}")
    print(s.round(1).to_string(index=False))
