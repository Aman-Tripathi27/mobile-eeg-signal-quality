"""Per-channel signal-quality checks (version 2).

The story (see reports/signal_quality_report.md): like a nurse doing rounds, we check every
sensor every 2 seconds on several "vital signs". Each vital sign has a yellow (warning) and a red
(fail) line. Each reading becomes a worry level 0..1, and the worst worry decides.

Two kinds of problem, kept separate because they need different fixes:

  CONTACT  - "can this sensor record at all?"      fix: repair / drop / interpolate the sensor
  ARTIFACT - "is this window polluted by something?" fix: clean or discard the window

Step 0 (scalp only): re-reference every sensor to the median of all sensors. Recorded against a
single reference sensor (FCz here), sensors next to it look artificially quiet and every sensor
inherits the reference's own hum. The median is barely affected by a few bad sensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, find_peaks, sosfiltfilt

BRAIN_BAND, LINE_BAND, HIGH_BAND = (1, 40), (59, 61), (30, 100)


@dataclass(frozen=True)
class Check:
    group: str      # "contact" or "artifact"
    warn: float     # yellow line -> worry 0.5
    fail: float     # red line    -> worry 1.0
    bad_when: str   # "high", "low" or "far" (too far from the other sensors in either direction)
    question: str
    source: str     # where the two lines came from


CHECKS = {
    "flat": Check("contact", 0.30, 0.60, "high",
                  "Is the signal frozen? (share of readings equal to the previous one)",
                  "our data: healthy <=15%, dead sensor R4 40-70%"),
    "clipped": Check("contact", 0.003, 0.02, "high",
                     "Is the signal stuck at the recorder's ceiling? (share of readings in flat tops of >=3)",
                     "one 3-reading flat top = 0.3% of a window; 2% = 40 ms of lost signal (judgement)"),
    "line_noise": Check("contact", 0.10, 0.30, "high",
                        "How much 60 Hz mains hum, relative to the brain signal?",
                        "guess checked on data: healthy 0.002-0.05, AFz 0.66"),
    "neighbour_corr": Check("contact", 0.40, 0.20, "low",
                            "Does it resemble at least one of its 4 nearest sensors? (best |correlation|)",
                            "0.4 = PREP pipeline; healthy 0.7-0.9 in our data; 0.2 judgement"),
    "high_freq_z": Check("artifact", 3.0, 6.0, "high",
                         "Much buzzier (>30 Hz, muscle) than the other sensors? (robust z)",
                         "statistics: |z|>3 happens ~0.3% by chance; PREP uses 5"),
    "amplitude_z": Check("artifact", 3.0, 6.0, "far",
                         "Much bigger or smaller than the other sensors? (robust z of size)",
                         "statistics: |z|>3 happens ~0.3% by chance; PREP uses 5"),
    "pop_ratio": Check("artifact", 15.0, 30.0, "high",
                       "Any sudden jump? (biggest step between readings / typical step)",
                       "pure noise gives ~5 (99.9th pct 7.3); real pops in our data 16-900"),
    "motion_r2": Check("artifact", 0.30, 0.60, "high",
                       "How much of the signal is predictable from head motion? (0..1)",
                       "standing: 99th pct <=0.2; jogging: 25% of windows >0.6"),
    "blink": Check("artifact", 0.30, 0.60, "high",
                   "Is a blink visible on this sensor? (blink size here / size on the forehead)",
                   "detector graded against eye sensors: finds ~90% of blinks standing; 0.3/0.6 judgement"),
}

# Blink detection uses scalp sensors only (Sabi's cap has no eye sensors).
BLINK_FRONT, BLINK_BACK = ("Fp1", "Fp2"), ("Pz", "POz", "Oz")
BLINK_THRESHOLD = 4.0  # x typical swing; chosen on sub-01..03, held up on sub-04..10
# Kept for older code/notebooks that read THRESHOLDS directly.
THRESHOLDS = {name: (c.warn, c.fail) for name, c in CHECKS.items()}


# ----------------------------------------------------------------------------- small helpers

def median_reference(data: np.ndarray) -> np.ndarray:
    """Express every channel relative to the median of all channels at each moment."""
    return data - np.median(data, axis=0, keepdims=True)


def _robust_z(x: np.ndarray) -> np.ndarray:
    """z-score across channels (axis 0) using median and MAD, so bad channels can't hide themselves."""
    med = np.median(x, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True) + 1e-12
    return (x - med) / mad


def bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """(channels, samples) -> (channels, windows, n) dropping the incomplete tail."""
    n_win = x.shape[-1] // n
    return x[..., : n_win * n].reshape(x.shape[0], n_win, n)


def _spectrum(w: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[-1]), axis=-1)) ** 2
    return np.fft.rfftfreq(w.shape[-1], 1 / sfreq), spec


def neighbours(positions: np.ndarray, k: int = 4) -> np.ndarray:
    """Index of the k nearest sensors (by 3-D position) for each sensor."""
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def _flat_top_mask(data: np.ndarray, ceiling: np.ndarray, min_run: int = 3) -> np.ndarray:
    """True where a reading is part of >= min_run consecutive readings at the ceiling.

    A healthy signal can touch its maximum once; a clipped one sits there (a flat top).
    """
    at = np.abs(data) >= 0.999 * ceiling[:, None]
    start = np.ones_like(at[:, : at.shape[1] - min_run + 1])
    for k in range(min_run):
        start &= at[:, k: at.shape[1] - min_run + 1 + k]
    member = np.zeros_like(at)
    for k in range(min_run):
        member[:, k: at.shape[1] - min_run + 1 + k] |= start
    return member


def _motion_r2(brain: np.ndarray, motion: np.ndarray, sfreq: float, n: int,
               max_lag_s: float = 0.05, lag_step_s: float = 0.01) -> np.ndarray:
    """Share (0..1) of each window's brain-band signal predictable from head motion.

    One linear filter per sensor (motion channels x small time lags) is learned on half the
    windows and scored on the other half, then swapped. Learning on the same 2 s it scores
    would overfit (version 1 of this check claimed 20-45% motion while standing still).
    """
    lags = np.arange(-int(max_lag_s * sfreq), int(max_lag_s * sfreq) + 1, max(1, int(lag_step_s * sfreq)))
    design = np.vstack([np.roll(m, lag) for m in motion for lag in lags])
    design -= design.mean(1, keepdims=True)
    y = brain - brain.mean(1, keepdims=True)
    n_win = y.shape[1] // n
    idx = np.arange(n_win * n).reshape(n_win, n)
    pred = np.zeros((y.shape[0], n_win * n))
    for k in (0, 1):
        train, test = idx[k::2].ravel(), idx[1 - k::2].ravel()
        beta, *_ = np.linalg.lstsq(design[:, train].T, y[:, train].T, rcond=None)
        pred[:, test] = (design[:, test].T @ beta).T
    yw, rw = _windows(y[:, : n_win * n], n), _windows(y[:, : n_win * n] - pred, n)
    return np.clip(1 - (rw ** 2).sum(-1) / ((yw ** 2).sum(-1) + 1e-30), 0, 1)


def detect_blinks(data: np.ndarray, names: list[str], sfreq: float,
                  threshold: float = BLINK_THRESHOLD) -> np.ndarray:
    """Blink times (sample indices) from scalp sensors only.

    A blink's fingerprint: a big slow swing on BOTH forehead sensors, in the same direction, and
    much smaller at the back of the head. Filtering turns a blink into an up- and a down-swing
    ~0.2 s apart, so we keep only the biggest swing within any 0.5 s (one event per blink).
    `data` should already be re-referenced (median reference).
    """
    rows = [names.index(c) for c in BLINK_FRONT + BLINK_BACK]
    band = bandpass(data[rows], sfreq, 1, 10)
    front, back = band[:2].mean(0), band[2:].mean(0)
    spread = 1.4826 * np.median(np.abs(front - np.median(front)))  # typical swing
    peaks, _ = find_peaks(np.abs(front), height=threshold * spread, distance=int(0.5 * sfreq))
    both_sides = np.sign(band[0, peaks]) == np.sign(band[1, peaks])
    fades_backwards = np.abs(back[peaks]) < 0.3 * np.abs(front[peaks])
    return peaks[both_sides & fades_backwards]


def blink_reach(data: np.ndarray, names: list[str], sfreq: float, blinks: np.ndarray,
                half_s: float = 0.25) -> np.ndarray:
    """How big this person's blinks are on each sensor, relative to the forehead (1 = as big)."""
    h = int(half_s * sfreq)
    ok = blinks[(blinks >= h) & (blinks < data.shape[1] - h)]
    if len(ok) < 5:
        return np.zeros(len(data))
    band = bandpass(data, sfreq, 1, 10)
    average_blink = np.mean([band[:, p - h:p + h] for p in ok], axis=0)  # (channels, time)
    size = np.abs(average_blink).max(-1)
    return np.clip(size / size[[names.index(c) for c in BLINK_FRONT]].mean(), 0, 1)


# ----------------------------------------------------------------------------- MEASURE

def window_features(data: np.ndarray, sfreq: float, positions: np.ndarray | None = None,
                    win_s: float = 2.0, rereference: bool = False,
                    ceiling: np.ndarray | None = None,
                    motion: np.ndarray | None = None,
                    names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Measure every check for every (channel, window). Returns arrays shaped (channels, windows).

    data       : (channels, samples) in volts, unfiltered
    positions  : (channels, 3) sensor positions -> enables the neighbour check
    rereference: apply the median reference first (use for scalp sensors)
    ceiling    : (channels,) largest storable value in volts; defaults to each channel's own
                 maximum (correct for this dataset, whose export scaled every channel to its max)
    motion     : (k, samples) head-motion signals on the same timeline -> enables motion_r2
    names      : channel names; if Fp1, Fp2, Pz, POz, Oz are present -> enables the blink check
    """
    n = int(win_s * sfreq)
    # Hardware checks look at the data exactly as recorded.
    raw_w = _windows(data, n)
    if ceiling is None:
        ceiling = np.abs(data).max(-1)
    feats = {
        "flat": (np.diff(raw_w, axis=-1) == 0).mean(-1),
        "clipped": _windows(_flat_top_mask(data, ceiling), n).mean(-1),
    }
    if rereference:
        data = median_reference(data)

    # Pop: biggest jump between consecutive readings, relative to this window's typical jump.
    steps = np.abs(np.diff(_windows(data, n), axis=-1))
    feats["pop_ratio"] = steps.max(-1) / (np.median(steps, axis=-1) + 1e-15)

    f, spec = _spectrum(_windows(data - data.mean(-1, keepdims=True), n), sfreq)
    in_band = lambda lo, hi: (f >= lo) & (f <= hi)
    brain_pow = spec[..., in_band(*BRAIN_BAND)].sum(-1)
    feats["line_noise"] = spec[..., in_band(*LINE_BAND)].sum(-1) / brain_pow
    # Median over frequency bins ignores narrow spikes (60 Hz, equipment lines) and measures broadband buzz.
    hi = in_band(*HIGH_BAND)
    high_pow = np.median(spec[..., hi], axis=-1) * hi.sum()
    feats["high_freq_z"] = _robust_z(np.log10(high_pow / brain_pow + 1e-30))

    brain_full = bandpass(data, sfreq, *BRAIN_BAND)
    brain = _windows(brain_full, n)
    feats["amplitude_z"] = _robust_z(np.log10(brain.std(-1) + 1e-15))

    if positions is not None:
        nb = neighbours(positions)
        z = (brain - brain.mean(-1, keepdims=True)) / (brain.std(-1, keepdims=True) + 1e-15)
        corr = np.einsum("cwt,cnwt->cnw", z, z[nb]) / n  # (channels, k, windows)
        # A healthy sensor resembles at least one neighbour; one that resembles none is suspicious.
        feats["neighbour_corr"] = np.max(np.abs(corr), axis=1)

    if motion is not None:
        feats["motion_r2"] = _motion_r2(brain_full, bandpass(motion, sfreq, *BRAIN_BAND), sfreq, n)

    if names is not None and all(c in names for c in BLINK_FRONT + BLINK_BACK):
        blinks = detect_blinks(data, names, sfreq)
        has_blink = np.zeros(data.shape[1] // n, dtype=bool)
        window_of_blink = blinks // n
        has_blink[window_of_blink[window_of_blink < len(has_blink)]] = True
        feats["blink"] = blink_reach(data, names, sfreq, blinks)[:, None] * has_blink[None, :]
    return feats


# ----------------------------------------------------------------------------- JUDGE

def badness(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Worry level per check: 0 below the yellow line, 0.5 at it, 1 at/after the red line."""
    out = {}
    for name, value in feats.items():
        if name not in CHECKS:
            continue
        c = CHECKS[name]
        v, warn, fail = value, c.warn, c.fail
        if c.bad_when == "far":
            v = np.abs(v)
        elif c.bad_when == "low":  # flip so that bigger always means worse
            v, warn, fail = -v, -warn, -fail
        out[name] = np.clip(0.5 + 0.5 * (v - warn) / (fail - warn), 0, 1) * (v >= warn)
    return out


# ----------------------------------------------------------------------------- COMBINE

def group_scores(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scores (1 = clean, 0 = unusable) per group and overall. The worst check in a group decides."""
    b = badness(feats)
    out = {}
    for group in ("contact", "artifact"):
        members = [b[k] for k in b if CHECKS[k].group == group]
        if members:
            out[group] = 1.0 - np.max(np.stack(members), axis=0)
    out["overall"] = np.minimum.reduce(list(out.values()))
    return out


def quality_score(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Overall signal-quality index, 1 = clean, 0 = unusable: 1 - worst worry of all checks."""
    return 1.0 - np.max(np.stack(list(badness(feats).values())), axis=0)


def worst_check(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Name of the check with the highest worry for every (channel, window) -> the 'why'."""
    b = badness(feats)
    names = list(b)
    return np.array(names)[np.argmax(np.stack([b[k] for k in names]), axis=0)]


BROKEN_IF_CONTACT_BAD_PCT = 50.0  # a sensor failing contact in most windows needs fixing, not cleaning


def summary_table(names: list[str], feats: dict[str, np.ndarray], score: np.ndarray | None = None,
                  authors_bad: list[str] | None = None) -> pd.DataFrame:
    """One row per channel: typical value of each check, % of bad windows, and a verdict.

    flat/clipped use the mean (they often hit only some windows); the rest use the median.
    """
    df = pd.DataFrame({k: (np.mean(v, axis=1) if k in ("flat", "clipped") else np.median(v, axis=1))
                       for k, v in feats.items()}, index=names)
    g = group_scores(feats)
    score = g["overall"] if score is None else score
    df["median_score"] = np.median(score, axis=1)
    df["pct_windows_bad"] = 100 * (score < 0.5).mean(axis=1)
    if "contact" in g:
        df["pct_contact_bad"] = 100 * (g["contact"] < 0.5).mean(axis=1)
    if "artifact" in g:
        df["pct_artifact_bad"] = 100 * (g["artifact"] < 0.5).mean(axis=1)
    if "pct_contact_bad" in df:
        df["verdict"] = np.where(df["pct_contact_bad"] > BROKEN_IF_CONTACT_BAD_PCT, "broken", "ok")
    if authors_bad is not None:
        df["authors_label"] = ["bad" if n in authors_bad else "" for n in names]
    return df.sort_values("median_score")


# ----------------------------------------------------------------------------- whole-recording view

def motion_coupling(data: np.ndarray, head_acc: np.ndarray, sfreq: float,
                    fmax: float = 10.0) -> np.ndarray:
    """Per channel: mean coherence (0..1) with head acceleration between 0.5 and `fmax` Hz.

    Coherence measures how consistently two signals rise and fall together at each rhythm,
    regardless of their size. High values mean the sensor is tracking head movement.
    """
    nper = int(8 * sfreq)
    out = np.zeros(data.shape[0])
    for i, ch in enumerate(data):
        f, c = coherence(ch, head_acc, fs=sfreq, nperseg=nper)
        out[i] = c[(f >= 0.5) & (f <= fmax)].mean()
    return out
 