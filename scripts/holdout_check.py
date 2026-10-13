"""Score people the rules have NEVER seen (sub-16..20) with usability.py exactly as it is.

Run:  .venv/bin/python scripts/holdout_check.py      (~5 min)
No threshold is changed here. This is the confirmation step: do the results from sub-01..10 hold?
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
from scipy.stats import spearmanr

from eegqc import lee, usability as U, validate as V

mne.set_log_level("error")
OUT_R, OUT_F = ROOT / "reports" / "results" / "holdout", ROOT / "reports" / "figures"
OUT_R.mkdir(parents=True, exist_ok=True)
HOLDOUT = [f"sub-{k:02d}" for k in range(16, 21)]
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
SF = 500

rows, sensor_rows, heat = [], [], {}
for sub in HOLDOUT:
    for ses in SESSIONS:
        if not (lee.ROOT / "sourcedata" / sub / ses / "eeg" / f"{sub}_{ses}_task-ERP_eeg.vhdr").exists():
            print(sub, ses, "not in the dataset (skipped)", flush=True)  # sub-19/20 have no jogging session
            continue
        raw = lee.load_eeg(sub, ses)
        scalp = lee.scalp_channels(raw)
        imu = lee.load_imu_aligned(raw, sub, ses)
        pos = np.array([raw.info["chs"][raw.ch_names.index(c)]["loc"][:3] for c in scalp])
        rec = U.Recording(raw.get_data(picks=scalp), SF, scalp, pos,
                          ceiling=lee.recorder_limits(sub, ses, channels=scalp),
                          motion=imu.get_data(picks=V.MOTION_CHANNELS))
        starts, n = rec.windows()
        use, hw, room = rec.measure(starts, n)
        score = U.usability_score(use)                       # (sensors, windows), 1 = usable
        flags = U.hardware_flags(hw)
        worries = U.worry(use, U.USABILITY_CHECKS)
        names = list(worries)
        why = np.array(names)[np.argmax(np.stack([worries[k] for k in names]), axis=0)]
        heat[(sub, ses)] = (scalp, score)

        trials, y, onsets, times = V.erp_trials(raw, scalp)
        auc = V.auc(y, V.out_of_fold_proba(V.decoding_features(trials, times), y))
        blinks, key = rec.blinks, V.eye_answer_key(raw)
        found, real = V.match_events(blinks, key, SF)
        reasons = pd.Series(why[score < 0.5]).value_counts(normalize=True).mul(100).round(0)
        rows.append(dict(sub=sub, ses=ses, speed=lee.SPEEDS[ses],
                         unusable_pct=100 * (score < 0.5).mean(), mean_score=score.mean(),
                         trial_quality=V.trial_scores(score, onsets, SF).mean(), auc=auc,
                         room_hum=float(np.median(room)), room_warning=bool(np.median(room) > U.ROOM_HUM_WARN),
                         hum_flagged_sensors=" ".join(scalp[i] for i in np.where(flags["hum_z"].mean(1) > 0.5)[0]),
                         top_reasons=", ".join(f"{k} {v:.0f}%" for k, v in reasons.head(3).items()),
                         blinks_found_pct=100 * found, blinks_real_pct=100 * real,
                         key_blinks_per_min=len(key) / (raw.times[-1] / 60)))
        for i, ch in enumerate(scalp):
            sensor_rows.append(dict(sub=sub, ses=ses, sensor=ch, median_score=np.median(score[i]),
                                    unusable_pct=100 * (score[i] < 0.5).mean()))
        print(sub, ses, f"unusable {rows[-1]['unusable_pct']:.1f}%  AUC {auc:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_R / "holdout_recordings.csv", index=False)
sens = pd.DataFrame(sensor_rows)
sens.to_csv(OUT_R / "holdout_sensors.csv", index=False)

pd.set_option("display.width", 230)
print("\nSCORES PER RECORDING")
print(df[["sub", "speed", "unusable_pct", "mean_score", "auc", "room_warning", "hum_flagged_sensors", "top_reasons"]]
      .round(3).to_string(index=False))
print("\nWORST SENSORS (most often unusable, over all 20 recordings)")
print(sens.groupby("sensor").unusable_pct.mean().sort_values(ascending=False).head(6).round(1).to_string())
print("\nTEST 1 - does the score track decoding across speeds? (per person, Spearman)")
track = {s: spearmanr(g.trial_quality, g.auc)[0] for s, g in df.groupby("sub")}
print({k: round(v, 2) for k, v in track.items()}, "| positive in", sum(v > 0 for v in track.values()), "of", len(track))
print("\nTEST 2 - scalp-only blinks vs eye sensors (median % found / % real)")
print(df.groupby("speed", sort=False)[["blinks_found_pct", "blinks_real_pct", "key_blinks_per_min"]].median().round(1).to_string())
print("\nunusable % by speed (mean):", df.groupby("speed", sort=False).unusable_pct.mean().round(1).to_dict())

# Figure: the scores themselves - every sensor, every 2 s, standing vs jogging, for each unseen person
fig, axes = plt.subplots(len(HOLDOUT), 2, figsize=(12, 3.2 * len(HOLDOUT)), sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.12})
for r, sub in enumerate(HOLDOUT):
    for c, ses in enumerate(["ses-02", "ses-05"]):
        ax = axes[r, c]
        if (sub, ses) not in heat:
            ax.set_title(f"{sub} - {lee.SPEEDS[ses]}: not in the dataset", loc="left", fontsize=9)
            ax.axis("off")
            continue
        scalp, score = heat[(sub, ses)]
        im = ax.imshow(score, aspect="auto", cmap="Blues_r", vmin=0, vmax=1, interpolation="nearest",
                       extent=[0, score.shape[1] * 2 / 60, len(scalp) - 0.5, -0.5])
        pct = 100 * (score < 0.5).mean()
        ax.set_title(f"{sub} - {lee.SPEEDS[ses]}: {pct:.0f}% unusable", loc="left", fontsize=9)
        ax.set_yticks([0, 8, 16, 24, 31], [scalp[i] for i in [0, 8, 16, 24, 31]], fontsize=7)
        ax.grid(False)
axes[-1, 0].set_xlabel("minutes"); axes[-1, 1].set_xlabel("minutes")
fig.colorbar(im, ax=axes, label="usability score (1 = usable, 0 = unusable; dark = bad)", shrink=0.4)
fig.suptitle("Usability scores on people the rules never saw (each row = one sensor)", x=0.12, y=0.92, ha="left")
fig.savefig(OUT_F / "fig10_holdout_scores.png", dpi=110, bbox_inches="tight"); plt.close(fig)
