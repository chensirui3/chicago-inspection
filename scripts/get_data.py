"""Fetch the City of Chicago's data snapshot into data/ (never committed to git).

    python scripts/get_data.py           # city snapshot -> data/raw, Parquet -> data/processed
    python scripts/get_data.py --portal  # also the small portal files used by Extension B

Steps: shallow-fetch github.com/Chicago/food-inspections-evaluation at a pinned commit into
data/external/, copy DATA/*.Rds into data/raw/, verify SHA-256 checksums, then convert every
.Rds file to Parquet once (the 85 MB business-license file takes a few minutes).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspections import config, load  # noqa: E402


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def fetch_city_repo(dest: Path, commit: str = config.CITY_REPO_COMMIT) -> Path:
    """Shallow clone of the city's repo at exactly ``commit``."""
    if (dest / ".git").exists():
        try:
            if _git("rev-parse", "HEAD", cwd=dest) == commit:
                return dest
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=dest)
    _git("remote", "add", "origin", config.CITY_REPO_URL, cwd=dest)
    print(f"fetching {config.CITY_REPO_URL} @ {commit[:10]} (about 140 MB) ...", flush=True)
    _git("fetch", "-q", "--depth", "1", "origin", commit, cwd=dest)
    _git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_snapshot(repo: Path) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in config.CITY_DATA_SHA256.items():
        dst = config.RAW_DIR / name
        if not dst.exists() or sha256(dst) != expected:
            shutil.copy2(repo / "DATA" / name, dst)
        got = sha256(dst)
        if got != expected:
            raise SystemExit(f"checksum mismatch for {name}: {got} != {expected}")
    print(f"verified {len(config.CITY_DATA_SHA256)} files in {config.RAW_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--portal", action="store_true", help="also download Extension B portal data")
    ap.add_argument("--force", action="store_true", help="rebuild the Parquet cache")
    args = ap.parse_args()

    repo = fetch_city_repo(config.EXTERNAL_DIR / "food-inspections-evaluation")
    copy_snapshot(repo)
    for name in load.RDS_NAMES:
        t0 = time.time()
        out = load.convert_rds(name, force=args.force)
        print(f"  {name:45s} -> {out}  ({time.time() - t0:.0f}s)",
              flush=True)
    if args.portal:
        for key, path in load.fetch_extension_b_data().items():
            print(f"  portal {key:20s} -> {path}")
    print("done")


if __name__ == "__main__":
    main()
