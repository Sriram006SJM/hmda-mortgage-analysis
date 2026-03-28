"""
download.py — Downloads HMDA historic zip files (2007–2017) with resume support.
Skips files already fully downloaded.

Usage:
    python download.py                  # download all years
    python download.py --years 2015 2016 2017   # specific years
"""

import argparse
import sys
import time
from pathlib import Path

import requests

from config import YEARS, URL_TEMPLATE, raw_zip

HEADERS = {"User-Agent": "HMDA-Pipeline/1.0"}


def file_size_remote(url: str) -> int:
    r = requests.head(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def download_year(year: int, force: bool = False) -> bool:
    url = URL_TEMPLATE.format(year=year)
    dest = raw_zip(year)

    remote_size = file_size_remote(url)
    local_size = dest.stat().st_size if dest.exists() else 0

    if not force and local_size == remote_size:
        print(f"  [{year}] Already downloaded ({local_size / 1e9:.2f} GB). Skipping.")
        return True

    # Resume support
    resume_header = {}
    mode = "wb"
    if local_size > 0 and local_size < remote_size:
        resume_header = {"Range": f"bytes={local_size}-"}
        mode = "ab"
        print(f"  [{year}] Resuming from {local_size / 1e6:.0f} MB…")
    else:
        print(f"  [{year}] Downloading {remote_size / 1e9:.2f} GB…")

    start = time.time()
    with requests.get(url, headers={**HEADERS, **resume_header},
                      stream=True, timeout=120) as resp:
        resp.raise_for_status()
        downloaded = local_size
        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / remote_size * 100
                elapsed = time.time() - start
                speed = (downloaded - local_size) / elapsed / 1e6 if elapsed > 0 else 0
                print(
                    f"\r  [{year}] {pct:.1f}%  {downloaded/1e9:.2f}/{remote_size/1e9:.2f} GB"
                    f"  {speed:.1f} MB/s",
                    end="", flush=True,
                )

    print(f"\n  [{year}] Done in {time.time()-start:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Downloading HMDA data for years: {args.years}")
    for year in sorted(args.years):
        try:
            download_year(year, force=args.force)
        except Exception as e:
            print(f"\n  [{year}] ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
