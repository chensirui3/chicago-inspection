"""Extension C: one run of the nightly live-scoring job.

    python scripts/nightly.py                 # incremental fetch since the last run
    python scripts/nightly.py --full-refresh  # refetch everything from the portal
    python scripts/nightly.py --backtest      # also recompute the monthly forward backtest
    python scripts/nightly.py --publish-only  # rebuild the page from the last run's cache

Set SOCRATA_APP_TOKEN in the environment (a GitHub Actions secret in CI); never in code.
Exits non-zero when the portal data is stale or its schema changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inspections import live  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--full-refresh", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--publish-only", action="store_true",
                    help="rebuild the page and README artifacts from the last run's cache")
    args = ap.parse_args()
    if args.publish_only:
        print(json.dumps(live.publish_only(), indent=2, default=str))
        return
    try:
        summary = live.run_nightly(backtest=args.backtest, full_refresh=args.full_refresh)
    except live.StaleDataError as e:
        print(f"DATA CHECK FAILED: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
