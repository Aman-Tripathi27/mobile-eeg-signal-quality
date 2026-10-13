"""Remove footstep artifacts using only the head motion sensor (what Sabi's cap has).

Method: gait-template subtraction (Gwin et al. 2010, J Neurophysiol, "Removal of movement artifact
from high-density EEG recorded during walking and running").

  1. Find each step (heel strike) as a sharp peak in the head's vertical acceleration.
  2. Give every moment a step phase: 0% at one heel strike, 100% at the next.
  3. For each sensor, average the EEG over the previous ~30 steps, aligned by phase. That average is
     the footstep artifact: anything locked to the steps survives averaging, while brain responses
     to the flashes (not timed to steps) average out.
  4. Subtract that template from the current step.
  5. (optional) Regress out what's left using the head motion signals at a few time lags.

No model is trained: the template is an average, and the regression is redone per recording.
Note: with one head sensor we can't tell left from right foot, so the cycle used is one STEP
(heel strike to the next heel strike), not a full two-step stride.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .quality import bandpass

VERTICAL = "HaccX"  # reads ~-9.8 m/s^2 at rest in this dataset, i.e. it points along gravity


def step_rate(acc_vertical: np.ndarray, sfreq: float) -> float:
    """Dominant rhythm (Hz) of the head bounce between 1 and 4 Hz (= steps per second)."""
    x = acc_vertical - acc_vertical.mean()
    f = np.fft.rfftfreq(len(x), 1 / sfreq)
    p = np.abs(np.fft.rfft(x)) ** 2
    band = (f >= 1) & (f <= 4)
    top = float(f[band][np.argmax(p[band])])
    # Some people's head bounce is strongest at TWICE the step rate (a double bump per step).
    # If there's a strong rhythm at half the top one (and still >= 1 step/s), that's the real step rate.
    half = (f >= top / 2 - 0.1) & (f <= top / 2 + 0.1)
    if top / 2 >= 1.0 and p[half].max() >= 0.3 * p[band].max():
        return float(f[half][np.argmax(p[half])])
    return top


def heel_strikes(acc_vertical: np.ndarray, sfreq: float, min_bounce: float = 0.5) -> np.ndarray:
    """Sample index of every step, or an empty array if the person isn't stepping.

    Peaks are searched in both directions (which way a heel strike points depends on how the sensor
    is mounted); we keep the direction whose steps come at the most regular intervals.
    """
    x = bandpass(acc_vertical[None], sfreq, 0.5, 20)[0]
    if x.std() < min_bounce:  # head barely moving (m/s^2): standing still
        return np.array([], dtype=int)
    period = sfreq / step_rate(acc_vertical, sfreq)
    best, best_irregularity = np.array([], dtype=int), np.inf
    for sign in (1, -1):
        for prominence in (0.5, 0.3, 0.2):
            peaks, _ = find_peaks(sign * x, distance=int(0.6 * period), prominence=prominence * x.std())
            if len(peaks) < 10:
                continue
            intervals = np.diff(peaks)
            # Irregularity ignores a few outliers (rest breaks), and penalises missing/extra steps:
            # the typical interval should match the step rate seen in the spectrum.
            spread = np.median(np.abs(intervals - np.median(intervals))) / np.median(intervals)
            mismatch = abs(np.median(intervals) - period) / period
            if spread + mismatch < best_irregularity:
                best, best_irregularity = peaks, spread + mismatch
    return best


def step_phase(n_samples: int, strikes: np.ndarray) -> np.ndarray:
    """Phase 0..1 within the current step for every sample (NaN before the first / after the last strike)."""
    phase = np.full(n_samples, np.nan)
    for a, b in zip(strikes[:-1], strikes[1:]):
        phase[a:b] = np.arange(b - a) / (b - a)
    return phase


def _resample_steps(data: np.ndarray, strikes: np.ndarray, n_points: int) -> np.ndarray:
    """(sensors, samples) -> (steps, sensors, n_points): each step stretched to the same length."""
    out = np.empty((len(strikes) - 1, data.shape[0], n_points))
    for k, (a, b) in enumerate(zip(strikes[:-1], strikes[1:])):
        pos = a + np.linspace(0, b - a - 1, n_points)
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        hi = np.minimum(lo + 1, data.shape[1] - 1)
        out[k] = data[:, lo] * (1 - frac) + data[:, hi] * frac
    return out


def remove_footsteps(data: np.ndarray, strikes: np.ndarray, n_previous: int = 30, min_previous: int = 10,
                     n_points: int = 100) -> np.ndarray:
    """Subtract the average of the previous `n_previous` steps (phase-aligned) from each step.

    Uses only PAST steps, so it could run live on a device. Steps with fewer than `min_previous`
    earlier steps, and unusually short/long steps (a stumble, a pause), are left untouched.
    """
    clean = data.copy()
    if len(strikes) < min_previous + 2:
        return clean
    steps = _resample_steps(data, strikes, n_points)                    # (steps, sensors, points)
    lengths = np.diff(strikes)
    typical = np.median(lengths)
    csum = np.concatenate([np.zeros((1,) + steps.shape[1:]), np.cumsum(steps, axis=0)])
    for k in range(min_previous, len(lengths)):
        if not (0.67 * typical <= lengths[k] <= 1.5 * typical):
            continue
        first = max(0, k - n_previous)
        template = (csum[k] - csum[first]) / (k - first)               # (sensors, points)
        a, b = strikes[k], strikes[k + 1]
        grid = np.linspace(0, n_points - 1, b - a)
        lo = np.floor(grid).astype(int)
        frac = grid - lo
        hi = np.minimum(lo + 1, n_points - 1)
        clean[:, a:b] -= template[:, lo] * (1 - frac) + template[:, hi] * frac
    return clean


def regress_out_motion(data: np.ndarray, motion: np.ndarray, sfreq: float, max_lag_s: float = 0.05,
                       lag_step_s: float = 0.01, fold_s: float = 2.0) -> np.ndarray:
    """Subtract what each sensor shares with the head motion signals (a few time lags).

    Fitted on every other 2-s block and applied to the rest, then swapped, so no stretch is cleaned
    with a fit that saw it.
    """
    lags = np.arange(-int(max_lag_s * sfreq), int(max_lag_s * sfreq) + 1, max(1, int(lag_step_s * sfreq)))
    design = np.vstack([np.roll(m, lag) for m in motion for lag in lags])
    design -= design.mean(1, keepdims=True)
    y = data - data.mean(1, keepdims=True)
    block = (np.arange(y.shape[1]) // int(fold_s * sfreq)) % 2
    pred = np.zeros_like(y)
    for k in (0, 1):
        train, test = block == k, block != k
        beta, *_ = np.linalg.lstsq(design[:, train].T, y[:, train].T, rcond=None)
        pred[:, test] = (design[:, test].T @ beta).T
    return data - pred
