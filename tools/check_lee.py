"""Sanity-check downloaded participants before using them (the checks we first did by hand).

Usage:  .venv/bin/python tools/check_lee.py sub-01 sub-02 ...
For every recording: are the files complete, are all event markers there (300 ERP / 60 SSVEP),
do brain and motion recordings have the same length, how far do their clocks drift?
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "lee2021_mobile_bci" / "sourcedata"
EXPECTED_MARKERS = {"ERP": 300, "SSVEP": 60}


def header(vhdr: Path) -> dict:
    t = vhdr.read_text(encoding="utf-8", errors="replace")
    n_ch = int(re.search(r"^NumberOfChannels=(\d+)", t, re.M).group(1))
    n_pts = int(re.search(r"^DataPoints=(\d+)", t, re.M).group(1))
    sfreq = 1e6 / float(re.search(r"^SamplingInterval=([\d.]+)", t, re.M).group(1))
    data = vhdr.with_suffix(".eeg") if vhdr.with_suffix(".eeg").exists() else vhdr.with_suffix(".dat")
    complete = data.exists() and data.stat().st_size == n_ch * n_pts * 2  # 2 bytes per reading
    markers = [int(p) - 1 for p in re.findall(r"^Mk\d+=Stimulus,[^,]*,(\d+)", vhdr.with_suffix(".vmrk").read_text(errors="replace"), re.M)]
    return dict(sfreq=sfreq, seconds=n_pts / sfreq, complete=complete, markers=np.array(markers) / sfreq)


rows = []
for sub in sys.argv[1:]:
    for ses in [f"ses-{k:02d}" for k in range(1, 6)]:
        for task in ["ERP", "SSVEP"]:
            eeg = ROOT / sub / ses / "eeg" / f"{sub}_{ses}_task-{task}_eeg.vhdr"
            imu = ROOT / sub / ses / "imu" / f"{sub}_{ses}_task-{task}_imu.vhdr"
            if ses == "ses-01" and task == "SSVEP":
                continue  # the training block only has the letter task
            row = dict(sub=sub, ses=ses, task=task, eeg_present=eeg.exists(), imu_present=imu.exists())
            if eeg.exists():
                e = header(eeg)
                row.update(eeg_complete=e["complete"], minutes=round(e["seconds"] / 60, 1),
                           markers=len(e["markers"]), markers_ok=len(e["markers"]) == EXPECTED_MARKERS[task])
            if eeg.exists() and imu.exists():
                i = header(imu)
                row.update(imu_complete=i["complete"], length_diff_s=round(abs(i["seconds"] - e["seconds"]), 2))
                if len(i["markers"]) == len(e["markers"]) and len(e["markers"]):
                    d = (i["markers"] - e["markers"]) * 1000
                    row.update(clock_drift_ms=round(d[-1] - d[0], 1))
                else:
                    row.update(clock_drift_ms=np.nan, imu_markers=len(i["markers"]))
            rows.append(row)

df = pd.DataFrame(rows)
problems = df[~(df.eeg_present & df.imu_present & df.get("eeg_complete", True) & df.get("imu_complete", True)
               & df.get("markers_ok", True) & (df.get("length_diff_s", 0) < 1)
               & df.get("clock_drift_ms", 0).between(-40, 40))]
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("\nRecordings with a problem:" if len(problems) else "\nNo problems found.")
if len(problems):
    print(problems.to_string(index=False))
