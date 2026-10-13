"""Loader for the Lee et al. 2021 mobile EEG dataset (OSF r7s9b).

Reads the full-detail `sourcedata/` copy and fixes the problems we found when inspecting it:
  * IMU headers point to `*_imu.eeg`, but the file on disk is `*_imu.dat`  -> we read the IMU ourselves
  * EEG and IMU clocks drift ~13-25 ms over a session                      -> we align them using shared event markers
  * event onsets are in samples and the codes are undocumented             -> we convert and name them
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Defaults to <repo>/data/lee2021_mobile_bci; override with EEGQC_LEE_ROOT.
ROOT = Path(
    os.environ.get(
        "EEGQC_LEE_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "lee2021_mobile_bci",
    )
)

SPEEDS = {
    "ses-01": "training block (standing)",
    "ses-02": "standing (0 m/s)",
    "ses-03": "slow walk (0.8 m/s)",
    "ses-04": "fast walk (1.6 m/s)",
    "ses-05": "light jog (2.0 m/s)",
}

EOG_CHANNELS = ["HEOGL", "HEOGR", "VEOGU", "VEOGL"]
# Ear sensors are named L1..L10 / R1..R8 (cEEGrid strips behind each ear).
EAR_PATTERN = re.compile(r"^[LR]\d+$")

# ERP task: code 1 = non-target (240 trials), code 2 = target (60 trials). Inferred from the
# 20% target rate in the paper; confirm it by checking which one produces the P300.
ERP_EVENT_ID = {"non-target": 1, "target": 2}


def _paths(sub: str, ses: str, task: str, root: Path = ROOT):
    base = root / "sourcedata" / sub / ses
    stem = f"{sub}_{ses}_task-{task}"
    return base / "eeg" / f"{stem}_eeg.vhdr", base / "imu" / f"{stem}_imu.vhdr"


def _read_markers(vmrk: Path) -> np.ndarray:
    """Return 0-based sample positions of all 'Stimulus' markers."""
    text = vmrk.read_text(encoding="utf-8", errors="replace")
    return np.array([int(p) - 1 for p in re.findall(r"^Mk\d+=Stimulus,[^,]*,(\d+)", text, re.M)])


def scalp_channels(raw: mne.io.BaseRaw) -> list[str]:
    return [ch for ch in raw.ch_names if ch not in EOG_CHANNELS and not EAR_PATTERN.match(ch)]


def ear_channels(raw: mne.io.BaseRaw) -> list[str]:
    return [ch for ch in raw.ch_names if EAR_PATTERN.match(ch)]


def load_eeg(sub: str, ses: str, task: str = "ERP", root: Path = ROOT) -> mne.io.Raw:
    """Brain + eye + ear sensors at 500 Hz, with standard 10-20 positions for the scalp sensors."""
    vhdr, _ = _paths(sub, ses, task, root)
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names})
    # Ear sensors have no standard positions, so they get none (on_missing="ignore").
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.info["line_freq"] = 60.0  # recorded in Korea
    return raw


def recorder_limits(sub: str, ses: str, task: str = "ERP", channels: list[str] | None = None,
                    root: Path = ROOT) -> np.ndarray:
    """Largest value (in volts) each channel can store before it clips.

    The file stores every reading as a 16-bit whole number (-32768..32767) times a per-channel
    scale factor ("resolution", in µV) given in the header. So the ceiling is 32767 x resolution.
    """
    vhdr, _ = _paths(sub, ses, task, root)
    text = vhdr.read_text(encoding="utf-8", errors="replace")
    res = {name: float(r or 1.0) for name, r in re.findall(r"^Ch\d+=([^,]*),[^,]*,([^,]*),", text, re.M)}
    names = channels if channels is not None else list(res)
    return np.array([32767 * res[n] * 1e-6 for n in names])


def authors_bad_channels(sub: str, ses: str, task: str = "ERP", root: Path = ROOT) -> list[str]:
    """Channels the dataset authors marked 'bad' (treat as weak labels, not ground truth)."""
    vhdr, _ = _paths(sub, ses, task, root)
    tsv = pd.read_csv(str(vhdr).replace("_eeg.vhdr", "_channels.tsv"), sep="\t")
    return tsv.loc[tsv["status"] == "bad", "name"].tolist()


def load_imu_aligned(raw_eeg: mne.io.BaseRaw, sub: str, ses: str, task: str = "ERP",
                     root: Path = ROOT) -> mne.io.RawArray:
    """Motion sensors (head, left ankle, right ankle) put onto the EEG's 500 Hz timeline.

    The IMU ran on its own 128 Hz clock that drifts relative to the EEG. Both files contain the
    same 300 (ERP) / 60 (SSVEP) event markers, so we map IMU time -> EEG time through those
    markers and interpolate the motion signals at every EEG sample.
    """
    vhdr_eeg, vhdr_imu = _paths(sub, ses, task, root)
    text = vhdr_imu.read_text(encoding="utf-8", errors="replace")
    n_ch = int(re.search(r"^NumberOfChannels=(\d+)", text, re.M).group(1))
    sfreq_imu = 1e6 / float(re.search(r"^SamplingInterval=([\d.]+)", text, re.M).group(1))
    chans = re.findall(r"^Ch\d+=([^,]*),[^,]*,([^,]*),", text, re.M)
    names = [c[0] for c in chans]
    scale = np.array([float(c[1] or 1.0) for c in chans])
    data_file = vhdr_imu.with_suffix(".dat")  # the header wrongly says .eeg
    imu = np.fromfile(data_file, dtype="<i2").reshape(-1, n_ch).T * scale[:, None]

    t_eeg_mk = _read_markers(vhdr_eeg.with_suffix(".vmrk")) / raw_eeg.info["sfreq"]
    t_imu_mk = _read_markers(vhdr_imu.with_suffix(".vmrk")) / sfreq_imu
    if len(t_eeg_mk) != len(t_imu_mk):
        raise ValueError(f"marker count mismatch: EEG {len(t_eeg_mk)} vs IMU {len(t_imu_mk)}")

    # EEG time of every EEG sample -> corresponding IMU time (linear fit outside the marker range).
    t_eeg = raw_eeg.times
    slope, intercept = np.polyfit(t_eeg_mk, t_imu_mk, 1)
    t_imu_at_eeg = np.interp(t_eeg, t_eeg_mk, t_imu_mk)
    outside = (t_eeg < t_eeg_mk[0]) | (t_eeg > t_eeg_mk[-1])
    t_imu_at_eeg[outside] = slope * t_eeg[outside] + intercept

    t_imu = np.arange(imu.shape[1]) / sfreq_imu
    aligned = np.vstack([np.interp(t_imu_at_eeg, t_imu, ch) for ch in imu])
    info = mne.create_info(names, raw_eeg.info["sfreq"], ch_types="misc")
    out = mne.io.RawArray(aligned, info, verbose="error")
    out.info["description"] = (f"IMU aligned to EEG; clock drift over session = "
                               f"{(t_imu_mk - t_eeg_mk)[-1]*1000 - (t_imu_mk - t_eeg_mk)[0]*1000:+.1f} ms")
    return out


def erp_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """MNE events array (sample, 0, code) for the ERP task: 1 = non-target, 2 = target."""
    events, mapping = mne.events_from_annotations(raw, verbose="error")
    code_of = {v: int(re.search(r"(\d+)\s*$", k).group(1)) for k, v in mapping.items()
               if k.startswith("Stimulus")}
    keep = np.isin(events[:, 2], list(code_of))
    events = events[keep]
    events[:, 2] = [code_of[c] for c in events[:, 2]]
    return events
