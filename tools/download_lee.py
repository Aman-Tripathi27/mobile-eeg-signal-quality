"""Download participants of the Lee et al. 2021 mobile EEG dataset (OSF project r7s9b).

Usage:  python tools/download_lee.py sub-04 sub-05 ...
Downloads both copies (sub-XX/ and sourcedata/sub-XX/) into data/lee2021_mobile_bci/.
Files already present with the right size are skipped, so it is safe to re-run after an error.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "lee2021_mobile_bci"
API = "https://api.osf.io/v2/nodes/r7s9b/files/osfstorage/"


def retry(fn, *args):
    for attempt in range(6):
        try:
            return fn(*args)
        except Exception as err:  # OSF sometimes answers 500/429; wait and try again
            print(f"  retry {attempt + 1}: {err}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {args}")


def get(url):
    return retry(lambda: json.load(urllib.request.urlopen(url, timeout=120)))


def children(url):
    items = []
    while url:
        page = get(url)
        items += page["data"]
        url = page["links"].get("next")
    return items


def files_under(item):
    a = item["attributes"]
    if a["kind"] == "folder":
        for child in children(item["relationships"]["files"]["links"]["related"]["href"]):
            yield from files_under(child)
    else:
        yield a["materialized_path"].lstrip("/"), a["size"] or 0, item["links"]["download"]


def main(subjects):
    root = children(API)
    todo = []
    for item in root:
        name = item["attributes"]["name"]
        if name in subjects:
            todo += files_under(item)
        elif name == "sourcedata":
            for sub in children(item["relationships"]["files"]["links"]["related"]["href"]):
                if sub["attributes"]["name"] in subjects:
                    todo += files_under(sub)
    print(f"{len(todo)} files, {sum(s for _, s, _ in todo) / 1e6:.0f} MB", flush=True)

    def fetch(job):
        path, size, url = job
        dst = OUT / path
        if dst.exists() and dst.stat().st_size == size:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        retry(urllib.request.urlretrieve, url, str(dst) + ".part")
        os.replace(str(dst) + ".part", dst)
        print(f"ok {size / 1e6:7.1f} MB {path}", flush=True)

    # OSF is slow per file, not per byte, so several downloads at once help a lot.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, todo))
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
