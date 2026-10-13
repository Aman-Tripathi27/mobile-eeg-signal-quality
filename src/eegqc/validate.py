"""Does the quality score mean anything? Tests against the brain's own "spotted it!" response (P300).

The P300 is a positive bump ~250-500 ms after a rare target appears, strongest at the top-back of
the head (Pz). It's measured from the difference between targets and non-targets, which the
quality score never looks at, so it's an independent judge of whether a signal is usable.

Two tests:
  A. Within each sensor: is the P300 clearer in trials the score calls good than in trials it calls bad?
  B. Decoding: can a classifier tell targets from non-targets better in high-scoring trials than in
     low-scoring ones? (scored with out-of-fold predictions, so no trial is judged by a model that saw it)
"""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import lee, quality

P300_WINDOW = (0.25, 0.50)   # seconds after the flash
P300_SENSORS = ["Pz", "P3", "P4", "CP1", "CP2", "POz", "PO3", "PO4"]
TMIN, TMAX = -0.2, 0.8
MOTION_CHANNELS = ["HaccX", "HaccY", "HaccZ", "HgyroX", "HgyroY", "HgyroZ"]


def score_recording(sub: str, ses: str, task: str = "ERP", win_s: float = 2.0):
    """Load one recording and run every quality check on the scalp sensors."""
    raw = lee.load_eeg(sub, ses, task)
    scalp = lee.scalp_channels(raw)
    imu = lee.load_imu_aligned(raw, sub, ses, task)
    pos = np.array([raw.info["chs"][raw.ch_names.index(c)]["loc"][:3] for c in scalp])
    feats = quality.window_features(
        raw.get_data(picks=scalp), raw.info["sfreq"], positions=pos, win_s=win_s, rereference=True,
        ceiling=lee.recorder_limits(sub, ses, task, channels=scalp),
        motion=imu.get_data(picks=MOTION_CHANNELS), names=scalp)
    return raw, scalp, feats, quality.group_scores(feats)


def eye_answer_key(raw: mne.io.BaseRaw, threshold: float = 5.0) -> np.ndarray:
    """Blink times (samples) from the EYE sensors: the answer key for the scalp-only detector.

    Uses the vertical eye signal (above-eye minus below-eye sensor). The threshold is relative to
    the typical swing, not the biggest one: MNE's standard finder scales to the single biggest swing,
    and one large artifact then hides almost every blink (it reported 0-2 blinks/min for 3 people).
    """
    sfreq = raw.info["sfreq"]
    veog = (raw.get_data(picks=["VEOGU"]) - raw.get_data(picks=["VEOGL"]))[0]
    band = quality.bandpass(veog[None], sfreq, 1, 10)[0]
    spread = 1.4826 * np.median(np.abs(band - np.median(band)))
    peaks, _ = find_peaks(np.abs(band), height=threshold * spread, distance=int(0.5 * sfreq))
    return peaks


def match_events(found: np.ndarray, truth: np.ndarray, sfreq: float, tol_s: float = 0.25):
    """Share of true events found, and share of found events that are true (within +-0.25 s)."""
    if len(found) == 0 or len(truth) == 0:
        return np.nan, np.nan
    tol = tol_s * sfreq
    recall = np.mean([np.any(np.abs(found - t) <= tol) for t in truth])
    precision = np.mean([np.any(np.abs(truth - f) <= tol) for f in found])
    return float(recall), float(precision)


def erp_trials(raw: mne.io.BaseRaw, scalp: list[str], l_freq: float = 0.5, h_freq: float = 20.0):
    """Single trials (trials, sensors, times) on the median reference, labels (1 = target), onsets."""
    data = quality.median_reference(raw.get_data(picks=scalp))
    return erp_trials_from_array(data, raw.info["sfreq"], lee.erp_events(raw), scalp, l_freq, h_freq)


def erp_trials_from_array(data: np.ndarray, sfreq: float, events: np.ndarray, scalp: list[str],
                          l_freq: float = 0.5, h_freq: float = 20.0):
    """Same as erp_trials, for data that's already re-referenced (e.g. after footstep cleaning)."""
    rr = mne.io.RawArray(data, mne.create_info(scalp, sfreq, "eeg"), verbose="error")
    rr.filter(l_freq, h_freq, verbose="error")
    ep = mne.Epochs(rr, events, {"non-target": 1, "target": 2}, tmin=TMIN, tmax=TMAX,
                    baseline=(TMIN, 0), preload=True, verbose="error")
    onsets = ep.events[:, 0]
    return ep.get_data(), (ep.events[:, 2] == 2).astype(int), onsets, ep.times


def trial_scores(score: np.ndarray, onsets: np.ndarray, sfreq: float, win_s: float = 2.0) -> np.ndarray:
    """(trials, sensors): worst window score overlapping each trial's -0.2..0.8 s span."""
    n = int(win_s * sfreq)
    first = np.clip((onsets + int(TMIN * sfreq)) // n, 0, score.shape[1] - 1)
    last = np.clip((onsets + int(TMAX * sfreq)) // n, 0, score.shape[1] - 1)
    return np.minimum(score[:, first], score[:, last]).T


def p300_effect(trials: np.ndarray, y: np.ndarray, times: np.ndarray) -> float:
    """Cohen's d of mean amplitude in the P300 window: targets vs non-targets (one sensor)."""
    win = (times >= P300_WINDOW[0]) & (times <= P300_WINDOW[1])
    amp = trials[:, win].mean(-1)
    a, b = amp[y == 1], amp[y == 0]
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def decoding_features(trials: np.ndarray, times: np.ndarray, bin_s: float = 0.05) -> np.ndarray:
    """Average each sensor in 50 ms bins from 0 to 0.8 s -> one flat feature vector per trial."""
    edges = np.arange(0, TMAX + 1e-9, bin_s)
    bins = [trials[:, :, (times >= lo) & (times < lo + bin_s)].mean(-1) for lo in edges[:-1]]
    return np.concatenate(bins, axis=1)


def out_of_fold_proba(X: np.ndarray, y: np.ndarray, n_splits: int = 5, repeats: int = 5,
                      seed: int = 0) -> np.ndarray:
    """Probability of 'target' for every trial, predicted by a model that never saw that trial."""
    proba = np.zeros(len(y))
    for r in range(repeats):
        for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y):
            mu, sd = X[train].mean(0), X[train].std(0) + 1e-12
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit((X[train] - mu) / sd, y[train])
            proba[test] += clf.predict_proba((X[test] - mu) / sd)[:, 1] / repeats
    return proba


def loo_distance(trials: np.ndarray, y: np.ndarray, times: np.ndarray, sensors: list[int]) -> np.ndarray:
    """How far each trial is from its class's average response, with that trial left out (µV RMS).

    A continuous, model-free target: noisy trials sit far from the person's typical response.
    Uses 0-0.8 s after the flash on the given sensors.
    """
    win = (times >= 0) & (times <= TMAX)
    x = trials[:, sensors][:, :, win] * 1e6
    out = np.zeros(len(y))
    for c in (0, 1):
        idx = np.where(y == c)[0]
        total = x[idx].sum(0)
        for i in idx:
            mean_without = (total - x[i]) / (len(idx) - 1)
            out[i] = np.sqrt(((x[i] - mean_without) ** 2).mean())
    return out


def p_correct(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold probability the decoder gave to the CORRECT class (continuous, 0..1)."""
    return np.where(y == 1, proba, 1 - proba)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
