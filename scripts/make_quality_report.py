"""Reproduce every number and figure in reports/signal_quality_report.md.

Run from the project folder:  .venv/bin/python scripts/make_quality_report.py
Takes ~40 minutes for 10 people. Writes reports/results/*.csv and reports/figures/*.png.

Scope = what Sabi's cap has: 32 scalp sensors + the head motion sensor. Ear sensors are ignored.
Eye sensors are used ONLY as the answer key for grading the scalp-only blink detector.

People are split into two groups so we can see whether results hold up:
  TUNING = sub-01..03 (thresholds were chosen looking at these)
  NEW    = sub-04..10 (never looked at while choosing thresholds)
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
from scipy.stats import spearmanr, wilcoxon

from eegqc import lee, quality, validate as V

mne.set_log_level("error")
OUT_R, OUT_F = ROOT / "reports" / "results", ROOT / "reports" / "figures"
OUT_R.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)
for old in OUT_F.glob("*.png"):
    old.unlink()

TUNING = ["sub-01", "sub-02", "sub-03"]
NEW = [f"sub-{k:02d}" for k in range(4, 11)]
SUBS = TUNING + NEW
GROUP = {s: ("tuning (sub-01..03)" if s in TUNING else "new (sub-04..10)") for s in SUBS}
SESSIONS = ["ses-02", "ses-03", "ses-04", "ses-05"]
SPEED_LABEL = {"ses-02": "stand\n0", "ses-03": "slow walk\n0.8", "ses-04": "fast walk\n1.6", "ses-05": "jog\n2.0"}
# Checks that matter after the 0.5-20 Hz filter used for the brain response (hum and >30 Hz buzz are filtered out).
RELEVANT = ["flat", "clipped", "neighbour_corr", "amplitude_z", "pop_ratio", "motion_r2", "blink"]

# Reference palette (dataviz skill, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPEED_SEQ = ["#b7d3f2", "#78a9e3", "#2a78d6", "#174a8b"]  # one hue, light -> dark = slow -> fast
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a9a8a2", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2, "figure.dpi": 130})

rec_rows, sensor_rows, reason_rows, blink_rows, trialA_rows, decode_rows = [], [], [], [], [], []
heatmap_cache, blink_example = {}, None

for sub in SUBS:
    for ses in SESSIONS:
        print(sub, ses, flush=True)
        raw, scalp, feats, g = V.score_recording(sub, ses)
        sf = raw.info["sfreq"]
        authors = [c for c in lee.authors_bad_channels(sub, ses) if c in scalp]
        table = quality.summary_table(scalp, feats, authors_bad=authors)
        why = quality.worst_check(feats)
        overall = g["overall"]
        if sub == "sub-02" and ses in ("ses-02", "ses-05"):
            heatmap_cache[ses] = (scalp, g["contact"], g["artifact"])

        rec_rows.append(dict(sub=sub, group=GROUP[sub], ses=ses, speed=lee.SPEEDS[ses],
                             pct_bad=100 * (overall < 0.5).mean(),
                             pct_contact_bad=100 * (g["contact"] < 0.5).mean(),
                             pct_artifact_bad=100 * (g["artifact"] < 0.5).mean(),
                             median_motion_r2=np.median(feats["motion_r2"]),
                             ours_broken=" ".join(table.index[table.verdict == "broken"]),
                             authors_bad=" ".join(authors)))
        for ch, row in table.iterrows():
            sensor_rows.append(dict(sub=sub, ses=ses, sensor=ch, **row.to_dict()))
        for name in quality.CHECKS:
            reason_rows.append(dict(sub=sub, group=GROUP[sub], ses=ses, check=name,
                                    pct_of_all_windows=100 * ((overall < 0.5) & (why == name)).mean()))

        # --- blink detector (scalp only) graded against the eye sensors
        X = quality.median_reference(raw.get_data(picks=scalp))
        found, truth = quality.detect_blinks(X, scalp, sf), V.eye_answer_key(raw)
        recall, precision = V.match_events(found, truth, sf)
        minutes = raw.times[-1] / 60
        blink_rows.append(dict(sub=sub, group=GROUP[sub], ses=ses, key_per_min=len(truth) / minutes,
                               scalp_per_min=len(found) / minutes, found_pct=100 * recall, real_pct=100 * precision))
        if sub == "sub-03" and ses == "ses-02":
            blink_example = (X[[scalp.index("Fp1"), scalp.index("Fp2")]].mean(0),
                             (raw.get_data(picks=["VEOGU"]) - raw.get_data(picks=["VEOGL"]))[0], found, truth, sf)

        # --- validation against the brain's "spotted it!" response
        trials, y, onsets, times = V.erp_trials(raw, scalp)
        proba = V.out_of_fold_proba(V.decoding_features(trials, times), y)
        relevant_score = quality.quality_score({k: feats[k] for k in RELEVANT if k in feats})
        for score_name, score in [("all checks", overall), ("relevant checks", relevant_score)]:
            ts = V.trial_scores(score, onsets, sf)
            q = ts.mean(1)
            lo, hi = np.quantile(q, [1 / 3, 2 / 3])
            decode_rows.append(dict(score=score_name, sub=sub, group=GROUP[sub], ses=ses, auc_all=V.auc(y, proba),
                                    auc_best_third=V.auc(y[q >= hi], proba[q >= hi]),
                                    auc_worst_third=V.auc(y[q <= lo], proba[q <= lo]), mean_trial_quality=q.mean(),
                                    p300_d_Pz=V.p300_effect(trials[:, scalp.index("Pz")], y, times)))
            for ch in V.P300_SENSORS:
                i = scalp.index(ch)
                good = ts[:, i] >= 0.5
                count = lambda m, c: int((m & (y == c)).sum())
                if min(count(good, 1), count(~good, 1)) < 10 or min(count(good, 0), count(~good, 0)) < 40:
                    continue
                trialA_rows.append(dict(score=score_name, sub=sub, group=GROUP[sub], ses=ses, sensor=ch,
                                        d_good=V.p300_effect(trials[good, i], y[good], times),
                                        d_bad=V.p300_effect(trials[~good, i], y[~good], times)))

rec, sens, reasons, blinks = map(pd.DataFrame, (rec_rows, sensor_rows, reason_rows, blink_rows))
testA, dec = pd.DataFrame(trialA_rows), pd.DataFrame(decode_rows)
for name, df in [("recordings", rec), ("sensors", sens), ("reasons", reasons), ("blinks", blinks),
                 ("test_A_within_sensor", testA), ("test_B_decoding", dec)]:
    df.to_csv(OUT_R / f"{name}.csv", index=False)
for stale in ["ear.csv"]:
    (OUT_R / stale).unlink(missing_ok=True)

# ============================================================================ group comparisons
groups = {"tuning (sub-01..03)": TUNING, "new (sub-04..10)": NEW, "all 10": SUBS}

# Speed table: does the picture from 3 people hold for the 7 new ones?
d_all = dec[dec.score == "all checks"]
speed_rows = []
for gname, members in groups.items():
    for ses in SESSIONS:
        r = rec[rec["sub"].isin(members) & (rec.ses == ses)]
        d = d_all[d_all["sub"].isin(members) & (d_all.ses == ses)]
        b = blinks[blinks["sub"].isin(members) & (blinks.ses == ses)]
        speed_rows.append(dict(group=gname, speed=lee.SPEEDS[ses], pct_bad=r.pct_bad.mean(),
                               pct_contact_bad=r.pct_contact_bad.mean(), pct_artifact_bad=r.pct_artifact_bad.mean(),
                               motion_r2=r.median_motion_r2.mean(), p300_d_Pz=d.p300_d_Pz.mean(), auc=d.auc_all.mean(),
                               blink_found_pct=b.found_pct.median(), blink_real_pct=b.real_pct.median(),
                               key_blinks_per_min=b.key_per_min.median()))
by_speed = pd.DataFrame(speed_rows)
by_speed.to_csv(OUT_R / "summary_by_group_and_speed.csv", index=False)

# Validation statistics per group
val_rows = []
for gname, members in groups.items():
    for score_name in ["all checks", "relevant checks"]:
        a = testA[(testA.score == score_name) & testA["sub"].isin(members)]
        b = dec[(dec.score == score_name) & dec["sub"].isin(members)]
        across = [spearmanr(gr.mean_trial_quality, gr.auc_all)[0] for _, gr in b.groupby("sub")]
        val_rows.append(dict(
            group=gname, score=score_name,
            across_median_rho=np.nanmedian(across), across_positive=f"{int(np.sum(np.array(across) > 0))} of {len(across)}",
            B_best_beats_worst=f"{int((b.auc_best_third > b.auc_worst_third).sum())} of {len(b)}",
            B_p=wilcoxon(b.auc_best_third, b.auc_worst_third).pvalue,
            B_median_best=b.auc_best_third.median(), B_median_worst=b.auc_worst_third.median(),
            A_good_beats_bad=f"{int((a.d_good > a.d_bad).sum())} of {len(a)}",
            A_p=wilcoxon(a.d_good, a.d_bad).pvalue if len(a) > 5 else np.nan))
validation = pd.DataFrame(val_rows)
validation.to_csv(OUT_R / "validation_stats.csv", index=False)

# Agreement with the authors' labels
agree_rows = []
for gname, members in groups.items():
    tp = fp = fn = 0
    for _, r in rec[rec["sub"].isin(members)].iterrows():
        ours, theirs = set(str(r.ours_broken).split()) - {"nan"}, set(str(r.authors_bad).split()) - {"nan"}
        tp, fp, fn = tp + len(ours & theirs), fp + len(ours - theirs), fn + len(theirs - ours)
    agree_rows.append(dict(group=gname, both=tp, only_ours=fp, only_authors=fn))
agreement = pd.DataFrame(agree_rows)
agreement.to_csv(OUT_R / "authors_agreement.csv", index=False)

# Which sensors are most often broken / most often polluted (all 10 people)
sens["is_broken"] = sens.verdict == "broken"
worst_sensors = (sens.groupby("sensor").agg(times_broken=("is_broken", "sum"),
                                            mean_pct_artifact_bad=("pct_artifact_bad", "mean"))
                 .sort_values("times_broken", ascending=False))
worst_sensors.to_csv(OUT_R / "sensors_ranked.csv")

pd.set_option("display.width", 220)
print(by_speed.round(2).to_string())
print(validation.round(3).to_string())
print(agreement.to_string())
print(worst_sensors.head(10).round(1).to_string())

# ============================================================================ figures
x = np.arange(len(SESSIONS))
xt = [SPEED_LABEL[s] for s in SESSIONS]

def per_subject_lines(ax, df, col, ylabel):
    """Thin grey line per person, bold blue line = mean of all 10."""
    for sub in SUBS:
        v = df[df["sub"] == sub].set_index("ses").loc[SESSIONS, col].values
        ax.plot(x, v, color=MUTED, lw=1, alpha=0.8)
    mean = df.groupby("ses")[col].mean().loc[SESSIONS].values
    ax.plot(x, mean, color=SERIES[0], lw=2.5, marker="o", markersize=6, label="mean of 10 people")
    ax.set_xticks(x, xt)
    ax.set_ylabel(ylabel)

# Fig 1: why windows fail, by speed. flat + clipped folded together (both tiny) to stay within 8 colours.
reasons["reason"] = reasons.check.replace({"flat": "frozen or clipped", "clipped": "frozen or clipped"})
order = ["frozen or clipped", "line_noise", "neighbour_corr", "high_freq_z", "amplitude_z", "pop_ratio", "blink", "motion_r2"]
pivot = (reasons.groupby(["sub", "ses", "reason"]).pct_of_all_windows.sum().groupby(["ses", "reason"]).mean()
         .unstack()[order].loc[SESSIONS])
fig, ax = plt.subplots(figsize=(7.8, 4))
bottom = np.zeros(len(SESSIONS))
for color, name in zip(SERIES, order):
    ax.bar(x, pivot[name].values, bottom=bottom, color=color, width=0.6, label=name, edgecolor=SURFACE, linewidth=2)
    bottom += pivot[name].values
for xi, total in zip(x, bottom):
    ax.text(xi, total + 0.8, f"{total:.0f}%", ha="center", color=INK)
ax.set_xticks(x, xt); ax.set_xlabel("speed (m/s)"); ax.set_ylabel("% of sensor-windows scored bad")
ax.set_title("Why windows fail, by speed (mean of 10 people)", loc="left", color=INK)
ax.legend(title="worst check", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout(); fig.savefig(OUT_F / "fig1_why_bad_by_speed.png"); plt.close(fig)

# Fig 2: what movement does
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
per_subject_lines(axes[0], rec, "median_motion_r2", "share of signal predictable\nfrom head motion")
per_subject_lines(axes[1], d_all, "p300_d_Pz", "brain-response strength at Pz\n(effect size d)")
per_subject_lines(axes[2], d_all, "auc_all", "decoding AUC (0.5 = guessing)")
axes[2].axhline(0.5, color=INK2, lw=1, ls="--")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Faster movement: more motion in the signal, weaker brain response, worse decoding (grey = one person)",
             x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(OUT_F / "fig2_speed_effects.png"); plt.close(fig)

# Fig 3: validation
d_rel = dec[dec.score == "relevant checks"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
for sub in SUBS:
    s = d_rel[d_rel["sub"] == sub].set_index("ses").loc[SESSIONS]
    axes[0].plot(s.mean_trial_quality, s.auc_all, color=MUTED, lw=1, zorder=1)
for ses, color in zip(SESSIONS, SPEED_SEQ):
    s = d_rel[d_rel.ses == ses]
    axes[0].scatter(s.mean_trial_quality, s.auc_all, color=color, s=40, zorder=2, edgecolor=SURFACE, linewidth=1.5,
                    label=lee.SPEEDS[ses])
axes[0].set_xlabel("mean quality score of the recording's trials"); axes[0].set_ylabel("decoding AUC")
axes[0].set_title("Across speeds (grey line = one person)", loc="left", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=8)
for _, r in d_rel.iterrows():
    axes[1].plot([0, 1], [r.auc_worst_third, r.auc_best_third], color=MUTED, lw=0.8, alpha=0.7)
axes[1].plot([0, 1], [d_rel.auc_worst_third.median(), d_rel.auc_best_third.median()], color=SERIES[0], lw=2.5,
             marker="o", markersize=7, label="median of 40 recordings")
axes[1].set_xticks([0, 1], ["worst-scoring third\nof trials", "best-scoring third\nof trials"]); axes[1].set_xlim(-0.3, 1.3)
axes[1].set_ylabel("decoding AUC"); axes[1].legend(frameon=False)
axes[1].set_title("Within a recording (one grey line = one recording)", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(OUT_F / "fig3_validation.png"); plt.close(fig)

# Fig 4: what the pop and clipping checks catch (scalp sensors only)
fp2_clip = sens[(sens["sub"] == "sub-01") & (sens.ses == "ses-05") & (sens.sensor == "Fp2")].clipped.values
examples = [("sub-01", "ses-05", "POz", 392, "pops: contact breaking during jogging"),
            ("sub-03", "ses-02", "Fp1", 211, "a step: sudden shift in level while standing"),
            ("sub-03", "ses-05", "FC2", 349, "a pop on every footstep while jogging"),
            ("sub-01", "ses-05", "Fp2", None, "clipping: flat tops at the recorder's ceiling while jogging")]
fig, axes = plt.subplots(len(examples), 1, figsize=(9, 7))
for ax, (sub, ses, ch, w, title) in zip(axes, examples):
    r = lee.load_eeg(sub, ses)
    sc = lee.scalp_channels(r)
    if w is None:  # find the window where this sensor is most clipped
        raw_ch = r.get_data(picks=[ch])
        lim = lee.recorder_limits(sub, ses, channels=[ch])
        mask = quality._windows(quality._flat_top_mask(raw_ch, lim), 1000)[0].mean(-1)
        w = int(np.argmax(mask))
        seg = raw_ch[0, w * 1000:(w + 1) * 1000] * 1e6
        ax.axhline(lim[0] * 1e6, color=SERIES[7], lw=1, ls="--"); ax.axhline(-lim[0] * 1e6, color=SERIES[7], lw=1, ls="--")
        ax.text(0.01, lim[0] * 1e6, "ceiling", color=INK2, fontsize=7, va="bottom")
    else:
        seg = quality.median_reference(r.get_data(picks=sc))[sc.index(ch), w * 1000:(w + 1) * 1000] * 1e6
        seg = seg - np.median(seg)
    ax.plot(np.arange(len(seg)) / 500, seg, color=SERIES[0], lw=1)
    ax.set_title(f"{sub} {lee.SPEEDS[ses]} - {ch}: {title}", loc="left", fontsize=8.5, color=INK)
    ax.set_ylabel("µV")
axes[-1].set_xlabel("time within the 2-second window (s)")
fig.tight_layout(); fig.savefig(OUT_F / "fig4_pops_and_clipping.png"); plt.close(fig)

# Fig 5: contact vs artifact heatmaps, same person standing vs jogging
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for col, ses in enumerate(["ses-02", "ses-05"]):
    sc, contact, artifact = heatmap_cache[ses]
    for row, (score, name) in enumerate([(contact, "CONTACT (is the sensor working?)"),
                                         (artifact, "ARTIFACT (is the window polluted?)")]):
        ax = axes[row, col]
        im = ax.imshow(score, aspect="auto", cmap="Blues_r", vmin=0, vmax=1, interpolation="nearest",
                       extent=[0, score.shape[1] * 2 / 60, len(sc) - 0.5, -0.5])
        ax.set_title(f"{name} - sub-02 {lee.SPEEDS[ses]}", loc="left", fontsize=8.5, color=INK)
        ax.grid(False)
    axes[1, col].set_xlabel("minutes")
for ax in axes[:, 0]:
    ax.set_yticks(range(len(sc)), sc, fontsize=6.5)
fig.colorbar(im, ax=axes, label="score (1 = clean, 0 = unusable; dark = bad)", shrink=0.6)
fig.savefig(OUT_F / "fig5_contact_vs_artifact_sub02.png", bbox_inches="tight"); plt.close(fig)

# Fig 6: blink detection from scalp only, graded by the eye sensors
front, veog, found, truth, sf = blink_example
t0, t1 = int(60 * sf), int(80 * sf)
tt = np.arange(t0, t1) / sf
fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
axes[0].plot(tt, quality.bandpass(front[None], sf, 1, 10)[0][t0:t1] * 1e6, color=SERIES[0], lw=1)
for p in found[(found >= t0) & (found < t1)]:
    axes[0].axvline(p / sf, color=SERIES[1], lw=1.5, alpha=0.8)
axes[0].set_ylabel("forehead (Fp1+Fp2)\nµV"); axes[0].set_title("Scalp-only detector (orange = blink found)", loc="left", fontsize=9, color=INK)
axes[1].plot(tt, quality.bandpass(veog[None], sf, 1, 10)[0][t0:t1] * 1e6, color=INK2, lw=1)
for p in truth[(truth >= t0) & (truth < t1)]:
    axes[1].axvline(p / sf, color=SERIES[2], lw=1.5, alpha=0.8)
axes[1].set_ylabel("eye sensor\nµV"); axes[1].set_xlabel("time (s)")
axes[1].set_title("Answer key from the eye sensors (green = real blink)", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(OUT_F / "fig6_blinks_scalp_vs_eye.png"); plt.close(fig)
print("figures written to", OUT_F)
