"""Signal quality, version 4: two layers, tied to how the signal will be used.

  USABILITY (the default score)  - "can this stretch of signal be used for the analysis?"
      Every check runs on the SAME frequency band the analysis uses (default 0.5-20 Hz, the band
      our brain-response decoder sees). Problems the analysis filters away don't count.
      Checks: flat, clipped, size vs other sensors, unlike neighbours, sudden jumps, follows head
      motion, blink, muscle leaking into the top of the band (+ mains hum, only if the band contains it).

  HARDWARE FLAGS (diagnostic only, never lower usability)
      "what should someone fix?"  hum worse than the other sensors (reseat it), high-frequency muscle,
      flat, clipped, and one ROOM warning per recording when mains hum is high everywhere
      (room wiring, chargers, or the shared reference/ground sensor).

Still rules only: fixed thresholds + the worry formula. Nothing is trained.
The band is a parameter: if a decoder uses higher frequencies, muscle and hum start counting
automatically, because they then fall inside the band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .quality import _flat_top_mask, _robust_z, bandpass, median_reference, neighbours

DEFAULT_BAND = (0.5, 20.0)
BLINK_FRONT, BLINK_BACK = ("Fp1", "Fp2"), ("Pz", "POz", "Oz")


@dataclass(frozen=True)
class Check:
    warn: float     # yellow line -> worry 0.5
    fail: float     # red line    -> worry 1.0
    bad_when: str   # "high", "low", or "far" (too far from the other sensors, either direction)
    question: str


USABILITY_CHECKS = {
    "flat": Check(0.30, 0.60, "high", "Frozen? (share of readings equal to the previous one)"),
    "clipped": Check(0.003, 0.02, "high", "Stuck at the recorder's ceiling? (signal lost at ALL frequencies)"),
    "amplitude_z": Check(3.0, 6.0, "far", "In-band size far from the other sensors? (robust z)"),
    "neighbour_corr": Check(0.40, 0.20, "low", "In-band signal like at least one of its 4 neighbours?"),
    "pop_ratio": Check(15.0, 30.0, "high", "Sudden jump? (biggest step / typical step, on 0.5-45 Hz)"),
    "motion_r2": Check(0.30, 0.60, "high", "Share of the in-band signal predictable from head motion"),
    "blink": Check(0.30, 0.60, "high", "Blink visible on this sensor? (size here / size on forehead)"),
    "muscle_edge_z": Check(3.0, 6.0, "high", "Muscle leaking into the top 5 Hz of the band, vs other sensors"),
    "line_in_band": Check(0.10, 0.30, "high", "Mains hum inside the band (only if the band contains 50/60 Hz)"),
}

HARDWARE_CHECKS = {
    "hum_z": Check(3.5, 6.0, "high", "More mains hum than the other sensors? (robust z, average reference)"),
    "muscle_z": Check(3.0, 6.0, "high", "More 30-100 Hz muscle buzz than the other sensors? (robust z)"),
    "flat": USABILITY_CHECKS["flat"],
    "clipped": USABILITY_CHECKS["clipped"],
}
ROOM_HUM_WARN = 1.0  # hum power >= brain (1-40 Hz) power on the typical sensor, as recorded


# ----------------------------------------------------------------------------- helpers

def detect_line_freq(data: np.ndarray, sfreq: float) -> int:
    """50 or 60 Hz: whichever peak stands out more above its neighbouring frequencies."""
    seg = data[:, : int(60 * sfreq)] - data[:, : int(60 * sfreq)].mean(-1, keepdims=True)
    f = np.fft.rfftfreq(seg.shape[1], 1 / sfreq)
    p = np.median(np.abs(np.fft.rfft(seg, axis=-1)) ** 2, axis=0)
    peakiness = {}
    for lf in (50, 60):
        peak = p[(f >= lf - 1) & (f <= lf + 1)].mean()
        around = p[((f >= lf - 4) & (f <= lf - 2)) | ((f >= lf + 2) & (f <= lf + 4))].mean()
        peakiness[lf] = peak / around
    return max(peakiness, key=peakiness.get)


def _segments(x: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """(channels, samples) -> (channels, segments, n) for arbitrary segment start samples."""
    idx = starts[:, None] + np.arange(n)[None, :]
    return x[:, idx]


def _power(seg: np.ndarray, sfreq: float):
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[-1]), axis=-1)) ** 2
    return np.fft.rfftfreq(seg.shape[-1], 1 / sfreq), spec


def detect_blinks(data_ref: np.ndarray, names: list[str], sfreq: float, threshold: float = 4.0,
                  min_s: float = 0.0, max_s: float = np.inf) -> np.ndarray:
    """Scalp-only blink times (samples), checking shape and location, not just size.

    A blink: big swing on both forehead sensors (> threshold x typical swing), same direction on
    left and right, and much smaller at the back of the head.
    A duration rule (100-400 ms) was tried and is OFF by default: on the filtered signal, 39-49% of
    real blinks measure under 100 ms, and real blinks and footstep mistakes have the same duration
    while jogging (median 101 vs 99 ms, sub-01..10), so it removed real blinks without removing errors.
    """
    rows = [names.index(c) for c in BLINK_FRONT + BLINK_BACK]
    band = bandpass(data_ref[rows], sfreq, 1, 10)
    front, back = band[:2].mean(0), band[2:].mean(0)
    spread = 1.4826 * np.median(np.abs(front - np.median(front)))
    peaks, _ = find_peaks(np.abs(front), height=threshold * spread, distance=int(0.5 * sfreq))
    if len(peaks) == 0:
        return peaks
    width_s = peak_widths(np.abs(front), peaks, rel_height=0.5)[0] / sfreq
    keep = ((np.sign(band[0, peaks]) == np.sign(band[1, peaks]))
            & (np.abs(back[peaks]) < 0.3 * np.abs(front[peaks]))
            & (width_s >= min_s) & (width_s <= max_s))
    return peaks[keep]


def blink_reach(data_ref: np.ndarray, names: list[str], sfreq: float, blinks: np.ndarray,
                half_s: float = 0.25) -> np.ndarray:
    """How big this person's blinks are on each sensor, relative to the forehead (1 = as big)."""
    h = int(half_s * sfreq)
    ok = blinks[(blinks >= h) & (blinks < data_ref.shape[1] - h)]
    if len(ok) < 5:
        return np.zeros(len(data_ref))
    band = bandpass(data_ref, sfreq, 1, 10)
    size = np.abs(np.mean([band[:, p - h:p + h] for p in ok], axis=0)).max(-1)
    return np.clip(size / size[[names.index(c) for c in BLINK_FRONT]].mean(), 0, 1)


def _motion_prediction(inband: np.ndarray, motion_inband: np.ndarray, sfreq: float,
                       max_lag_s: float = 0.05, lag_step_s: float = 0.01, fold_s: float = 2.0) -> np.ndarray:
    """Part of each sensor's in-band signal predictable from head motion (a few time lags).

    Learned on every other 2-s block and predicted on the rest, then swapped, so no stretch is
    predicted by a fit that saw it. A measurement redone per recording; nothing is kept or learned.
    """
    lags = np.arange(-int(max_lag_s * sfreq), int(max_lag_s * sfreq) + 1, max(1, int(lag_step_s * sfreq)))
    design = np.vstack([np.roll(m, lag) for m in motion_inband for lag in lags])
    design -= design.mean(1, keepdims=True)
    y = inband - inband.mean(1, keepdims=True)
    block = (np.arange(y.shape[1]) // int(fold_s * sfreq)) % 2
    pred = np.zeros_like(y)
    for k in (0, 1):
        train, test = block == k, block != k
        beta, *_ = np.linalg.lstsq(design[:, train].T, y[:, train].T, rcond=None)
        pred[:, test] = (design[:, test].T @ beta).T
    return pred


# ----------------------------------------------------------------------------- prepare once, score any segments

class Recording:
    """Everything computed once per recording; then any set of segments can be scored quickly.

    data      : (sensors, samples) in volts, as recorded (scalp sensors only)
    names     : sensor names;  positions: (sensors, 3)
    ceiling   : (sensors,) largest storable value in volts (for clipping)
    motion    : (k, samples) head motion signals on the same timeline, or None
    band      : the analysis band (default 0.5-20 Hz)
    """

    def __init__(self, data, sfreq, names, positions, ceiling=None, motion=None, band=DEFAULT_BAND):
        self.sfreq, self.names, self.band = sfreq, list(names), band
        self.raw = data
        self.ceiling = np.abs(data).max(-1) if ceiling is None else ceiling
        self.flat_top = _flat_top_mask(data, self.ceiling)
        self.line_freq = detect_line_freq(data, sfreq)
        self.ref = median_reference(data)                   # for the usability layer
        self.avg_ref = data - data.mean(0, keepdims=True)    # for the hum comparison (shared hum cancels)
        self.inband = bandpass(self.ref, sfreq, *band)
        # Pops are recognised by their sharpness, which a 20 Hz filter blurs into blink-like swings.
        # So pops are detected up to 45 Hz (kept below the mains frequency); a pop still damages the
        # in-band signal, so it still counts against usability.
        self.sharp = bandpass(self.ref, sfreq, 0.5, min(45.0, self.line_freq - 5.0))
        self.nb = neighbours(positions)
        self.motion_pred = None
        if motion is not None:
            self.motion_pred = _motion_prediction(self.inband, bandpass(motion, sfreq, *band), sfreq)
        self.blinks = np.array([], dtype=int)
        self.reach = np.zeros(len(names))
        if all(c in names for c in BLINK_FRONT + BLINK_BACK):
            self.blinks = detect_blinks(self.ref, self.names, sfreq)
            self.reach = blink_reach(self.ref, self.names, sfreq, self.blinks)

    # -- segment helpers
    def windows(self, win_s: float = 2.0) -> tuple[np.ndarray, int]:
        n = int(win_s * self.sfreq)
        return np.arange(self.raw.shape[1] // n) * n, n

    def measure(self, starts: np.ndarray, n: int) -> tuple[dict, dict, np.ndarray]:
        """Usability measurements, hardware measurements (each (sensors, segments)), room hum per segment."""
        sf, lo, hi = self.sfreq, *self.band
        raw_s, ib = _segments(self.raw, starts, n), _segments(self.inband, starts, n)
        use = {"flat": (np.diff(raw_s, axis=-1) == 0).mean(-1),
               "clipped": _segments(self.flat_top, starts, n).mean(-1)}
        use["amplitude_z"] = _robust_z(np.log10(ib.std(-1) + 1e-15))
        z = (ib - ib.mean(-1, keepdims=True)) / (ib.std(-1, keepdims=True) + 1e-15)
        use["neighbour_corr"] = np.max(np.abs(np.einsum("cwt,cnwt->cnw", z, z[self.nb]) / n), axis=1)
        steps = np.abs(np.diff(_segments(self.sharp, starts, n), axis=-1))
        use["pop_ratio"] = steps.max(-1) / (np.median(steps, axis=-1) + 1e-15)
        if self.motion_pred is not None:
            resid = ib - _segments(self.motion_pred, starts, n)
            ibc = ib - ib.mean(-1, keepdims=True)
            use["motion_r2"] = np.clip(1 - (resid - resid.mean(-1, keepdims=True)).var(-1) / (ibc.var(-1) + 1e-30), 0, 1)
        has_blink = np.zeros(len(starts), dtype=bool)
        for b in self.blinks:
            has_blink |= (b >= starts) & (b < starts + n)
        use["blink"] = self.reach[:, None] * has_blink[None, :]

        ref_s = _segments(self.ref, starts, n)
        f, spec = _power(ref_s - ref_s.mean(-1, keepdims=True), sf)
        del ref_s
        band_pow = spec[..., (f >= lo) & (f <= hi)].sum(-1) + 1e-30
        edge = (f >= max(lo, hi - 5)) & (f <= hi)
        use["muscle_edge_z"] = _robust_z(np.log10(spec[..., edge].sum(-1) / band_pow + 1e-30))
        lf = self.line_freq
        if lo <= lf <= hi:
            use["line_in_band"] = spec[..., (f >= lf - 1) & (f <= lf + 1)].sum(-1) / band_pow

        # Hardware layer
        hw = {"flat": use["flat"], "clipped": use["clipped"]}
        avg_s = _segments(self.avg_ref, starts, n)
        fa, spa = _power(avg_s - avg_s.mean(-1, keepdims=True), sf)
        del avg_s
        brain_pow = spa[..., (fa >= 1) & (fa <= 40)].sum(-1) + 1e-30
        hw["hum_z"] = _robust_z(np.log10(spa[..., (fa >= lf - 1) & (fa <= lf + 1)].sum(-1) / brain_pow + 1e-30))
        high = (fa >= 30) & (fa <= 100)
        hw["muscle_z"] = _robust_z(np.log10(np.median(spa[..., high], -1) * high.sum() / brain_pow + 1e-30))
        # Room: hum as recorded (before any re-referencing), typical sensor
        fr, spr = _power(raw_s - raw_s.mean(-1, keepdims=True), sf)
        room = np.median(spr[..., (fr >= lf - 1) & (fr <= lf + 1)].sum(-1)
                         / (spr[..., (fr >= 1) & (fr <= 40)].sum(-1) + 1e-30), axis=0)
        return use, hw, room


# ----------------------------------------------------------------------------- judge and combine

def worry(values: dict, checks: dict) -> dict:
    """Worry per check: 0 below the yellow line, 0.5 at it, rising to 1 at the red line."""
    out = {}
    for name, v in values.items():
        c = checks[name]
        val, warn, fail = v, c.warn, c.fail
        if c.bad_when == "far":
            val = np.abs(v)
        elif c.bad_when == "low":
            val, warn, fail = -v, -warn, -fail
        out[name] = np.clip(0.5 + 0.5 * (val - warn) / (fail - warn), 0, 1) * (val >= warn)
    return out


def severity(values: dict, checks: dict) -> dict:
    """How far past the yellow line, as log(value / yellow), 0 if not past. No ceiling, so no ties."""
    out = {}
    for name, v in values.items():
        c = checks[name]
        if c.bad_when == "high":
            ratio = np.maximum(v, 1e-12) / c.warn
        elif c.bad_when == "far":
            ratio = np.abs(v) / c.warn
        else:  # low is bad
            ratio = c.warn / np.maximum(v, 1e-12)
        out[name] = np.maximum(0.0, np.log(ratio))
    return out


def usability_score(use: dict) -> np.ndarray:
    """1 = usable, 0 = unusable: 1 - the worst worry (pass/fail style)."""
    return 1.0 - np.max(np.stack(list(worry(use, USABILITY_CHECKS).values())), axis=0)


def usability_severity(use: dict) -> np.ndarray:
    """0 = nothing past any yellow line; larger = further past, averaged over checks (for ranking)."""
    return np.mean(np.stack(list(severity(use, USABILITY_CHECKS).values())), axis=0)


def hardware_flags(hw: dict) -> dict:
    """True where a hardware check is past its yellow line (diagnostic, never lowers usability)."""
    return {k: v > 0.5 for k, v in worry(hw, HARDWARE_CHECKS).items()}
