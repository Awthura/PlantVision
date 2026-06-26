"""
Download PhenoBench v1.1.0 (~7.6 GB) into data/.

Usage:
    python scripts/download_data.py [--out data/PhenoBench]

IMPORTANT: Confirm the current download URL and checksum at phenobench.org/dataset.html
before running. The URL and MD5 below are placeholders — fill them in from the
official dataset page.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── UPDATE THESE before first run ──────────────────────────────────────────
DATASET_URL = "https://www.phenobench.org/data/PhenoBench-v110.zip"
EXPECTED_MD5 = ""                                      # set if checksum is published
ZIP_NAME = "PhenoBench-v110.zip"
# ───────────────────────────────────────────────────────────────────────────


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    import urllib.request

    log.info("Downloading %s → %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    log.info("Download complete (%s)", dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/PhenoBench", help="Extraction target dir")
    parser.add_argument("--url", default=DATASET_URL, help="Direct download URL")
    args = parser.parse_args()

    out_dir = Path(args.out)
    zip_path = out_dir.parent / ZIP_NAME

    if not zip_path.exists():
        download(args.url, zip_path)
    else:
        log.info("Zip already exists: %s", zip_path)

    if EXPECTED_MD5:
        actual = md5(zip_path)
        if actual != EXPECTED_MD5:
            raise ValueError(f"MD5 mismatch! expected={EXPECTED_MD5} got={actual}")
        log.info("MD5 OK")

    if not out_dir.exists():
        log.info("Extracting %s → %s …", zip_path.name, out_dir)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir.parent)
        log.info("Extraction complete")
    else:
        log.info("Already extracted: %s", out_dir)


if __name__ == "__main__":
    main()
