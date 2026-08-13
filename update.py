"""Sync the local mirror against the IACR ePrint archive.

Fetches by *gap* rather than by high-water mark: every id missing from the range
1..target is retried, so a paper skipped by a transient failure gets picked up on
a later run instead of being stranded behind the maximum id forever. Ids that
answer 404 are recorded in withdrawn.json so they are not re-requested on every
subsequent run.
"""

import argparse
import json
import os
import sys
import time

import requests
from lxml import etree

BASE = "https://eprint.iacr.org"
WITHDRAWN = "withdrawn.json"

# Highest id actually issued in a year, where that exceeds the archive's count of
# live papers (withdrawn papers still consume an id). Saves probing past the end
# of years we already know the shape of.
KNOWN_MAX_ID = {
    "2004": 377, "2005": 469, "2006": 486, "2009": 637, "2010": 661,
    "2013": 882, "2015": 1256, "2016": 1196, "2018": 1251, "2019": 1499,
    "2020": 1620, "2021": 1705, "2022": 1781, "2023": 1973, "2024": 2100,
    "2025": 2340,
}


class Throttled(Exception):
    """The archive asked us to stop. Carries Retry-After when one was sent."""

    def __init__(self, retry_after=None):
        self.retry_after = retry_after
        super().__init__(f"rate limited (Retry-After: {retry_after or 'unset'})")


def make_session(user_agent=None):
    s = requests.Session()
    # Left at the requests default unless EPRINT_UA is set. That default is what
    # this mirror has always sent, and it is at least an accurate statement of
    # what the client is. Set EPRINT_UA to identify the mirror explicitly.
    if user_agent:
        s.headers["User-Agent"] = user_agent
    return s


def peep_paper_counts(session) -> dict[str, int]:
    """Live paper count per year, scraped from the archive's index."""
    r = session.get(f"{BASE}/byyear", timeout=60)
    if r.status_code == 429:
        raise Throttled(r.headers.get("Retry-After"))
    r.raise_for_status()
    tree = etree.HTML(r.text)
    lis = tree.xpath("//h3[text()='By year']/following-sibling::ul[1]/li/a")
    counts = {li.text: int(li.tail.strip(" ()").split()[0]) for li in lis}
    # An error page parses fine and yields nothing, which would silently look
    # like "every year is already complete". Refuse to plan against that.
    if not counts:
        raise RuntimeError(f"no year counts found at {BASE}/byyear (HTTP {r.status_code})")
    return counts


def local_ids(year: str) -> set[int]:
    """Ids already on disk, in any format (a few early papers are .ps)."""
    if not os.path.isdir(year):
        return set()
    return {
        int(stem)
        for f in os.listdir(year)
        if (stem := f.split(".")[0]).isdigit()
    }


def verify_local(years, remove=True) -> list[tuple[str, str]]:
    """Find files that are not the format their name claims.

    A 404 page saved as .pdf still counts as "present" to gap detection, so it
    would never be re-fetched; 2013/133.pdf sat in the archive that way for
    years. Removing it turns it back into a gap the sync will fill.
    """
    bad = []
    for year in years:
        if not os.path.isdir(year):
            continue
        for f in sorted(os.listdir(year)):
            if f.startswith("."):
                continue
            path = os.path.join(year, f)
            magic = b"%!" if f.endswith(".ps") else b"%PDF"
            with open(path, "rb") as fh:
                head = fh.read(4)
            if os.path.getsize(path) == 0:
                bad.append((path, "empty"))
            elif not head.startswith(magic):
                bad.append((path, f"bad magic {head!r}"))
    if remove:
        for path, _ in bad:
            os.remove(path)
    return bad


def load_withdrawn() -> dict[str, set[int]]:
    if not os.path.exists(WITHDRAWN):
        return {}
    with open(WITHDRAWN) as f:
        return {y: set(v) for y, v in json.load(f).items()}


def save_withdrawn(w: dict[str, set[int]]):
    with open(WITHDRAWN, "w") as f:
        json.dump({y: sorted(v) for y, v in sorted(w.items()) if v}, f, indent=1)
        f.write("\n")


def get_item(session, year: str, ct: int, delay: float, retries: int) -> str:
    """Fetch one paper. Returns "ok", "gone" (404), or "fail" (retry later)."""
    url = f"{BASE}/{year}/{ct}.pdf"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120)
        except requests.RequestException as e:
            print(f"  {year}/{ct}: {type(e).__name__}, retrying")
            time.sleep(delay * 2 ** attempt)
            continue

        if r.status_code == 404:
            return "gone"
        if r.status_code == 429:
            # A rate limit is a statement about the whole run, not about this
            # one file. Retrying per-paper turns a throttle into thousands of
            # refused requests: the first version of this script ground out
            # 2551 of them over three hours and fetched nothing. Stop the run.
            raise Throttled(r.headers.get("Retry-After"))
        if r.status_code != 200:
            if r.status_code not in (500, 502, 503, 504):
                print(f"  {year}/{ct}: HTTP {r.status_code}")
                return "fail"
            print(f"  {year}/{ct}: HTTP {r.status_code}, backing off")
            time.sleep(delay * 2 ** attempt)
            continue

        # A WAF block or error page returns 200 with HTML. Never let one land in
        # the archive under a .pdf name.
        if not r.content.startswith(b"%PDF"):
            print(f"  {year}/{ct}: not a PDF ({r.content[:40]!r})")
            return "fail"

        # Write via a temp name so an interrupted run cannot leave a truncated file.
        tmp = f"{year}/.{ct}.pdf.tmp"
        with open(tmp, "wb") as f:
            f.write(r.content)
        os.replace(tmp, f"{year}/{ct}.pdf")
        return "ok"
    return "fail"


def targets(year: str, counts: dict[str, int]) -> int:
    """Highest id worth attempting for a year."""
    return max(counts.get(year, 0), KNOWN_MAX_ID.get(year, 0), max(local_ids(year), default=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", action="append", help="only sync these years")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--probe-ahead", type=int, default=5,
                    help="consecutive misses past the end before a year is done")
    ap.add_argument("--recheck-withdrawn", action="store_true",
                    help="re-request ids previously recorded as 404")
    ap.add_argument("--dry-run", action="store_true", help="report gaps, fetch nothing")
    ap.add_argument("--verify", action="store_true",
                    help="drop files whose contents are not the claimed format, then refetch")
    args = ap.parse_args()

    session = make_session(os.environ.get("EPRINT_UA"))
    counts = peep_paper_counts(session)
    withdrawn = {} if args.recheck_withdrawn else load_withdrawn()

    years = args.year or sorted(set(counts) | {d for d in os.listdir() if d.isdigit()})

    if args.verify:
        for path, why in verify_local(years, remove=not args.dry_run):
            print(f"corrupt: {path} ({why})")

    plan = {}
    for year in years:
        target = targets(year, counts)
        have = local_ids(year)
        skip = withdrawn.get(year, set())
        missing = [i for i in range(1, target + 1) if i not in have and i not in skip]
        if missing or year in counts:
            plan[year] = (missing, target)
        n_skip = len(skip)
        print(f"{year}\thave {len(have):<5} target {target:<5} missing {len(missing):<5}"
              + (f" (skipping {n_skip} withdrawn)" if n_skip else ""))

    total = sum(len(m) for m, _ in plan.values())
    print(f"\n{total} papers to fetch"
          + (f", plus probing up to {args.probe_ahead} past each year's end" if not args.dry_run else ""))
    if args.dry_run:
        return

    # Don't start a bulk run into a limiter that is already saying no.
    probe = session.get(f"{BASE}/robots.txt", timeout=60)
    if probe.status_code == 429:
        print("archive is currently rate limiting us; not starting.", file=sys.stderr)
        return 1
    time.sleep(args.delay)

    fetched = failed = 0
    try:
        for year, (missing, target) in plan.items():
            os.makedirs(year, exist_ok=True)

            for ct in missing:
                status = get_item(session, year, ct, args.delay, args.retries)
                if status == "ok":
                    fetched += 1
                    print(f"{year}/{ct}.pdf  ({fetched}/{total})")
                elif status == "gone":
                    withdrawn.setdefault(year, set()).add(ct)
                else:
                    failed += 1
                time.sleep(args.delay)

            # The index counts live papers, so the true last id can sit past it.
            # Walk forward until the archive has been quiet for a few ids.
            ct, misses = target + 1, 0
            while misses < args.probe_ahead:
                if ct in local_ids(year):
                    ct += 1
                    continue
                status = get_item(session, year, ct, args.delay, args.retries)
                if status == "ok":
                    fetched += 1
                    misses = 0
                    print(f"{year}/{ct}.pdf  (past index)")
                else:
                    misses += 1
                ct += 1
                time.sleep(args.delay)
    except Throttled as t:
        print(f"\n{t}. Stopping so the archive is not hammered while it is "
              f"asking us to back off; rerun later to resume where this left off.",
              file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        save_withdrawn(withdrawn)
        print(f"\nfetched {fetched}, failed {failed}, "
              f"{sum(len(v) for v in withdrawn.values())} ids recorded withdrawn")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Throttled as t:
        print(f"{t}. The archive is asking us to back off; try again later.",
              file=sys.stderr)
        sys.exit(1)
