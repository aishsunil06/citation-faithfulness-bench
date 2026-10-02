#!/usr/bin/env python3
"""Capture plain-text article prose from Wikipedia into a raw JSONL file.

This lives in `tools/` rather than `cfbench/` on purpose: the library itself
must stay network-free and dependency-free, so corpus capture is an operator
step whose output is a plain `{url, title, raw}` file. `cfbench ingest` then
cleans and dedupes it offline, which also means a run is reproducible from the
committed corpus without touching the network again.

Wikipedia is the source because it offers what a citation-faithfulness corpus
needs and most free text does not: dense factual prose with figures, dates and
named entities, stable URLs, a permissive licence, and enough topical overlap
between related articles to produce genuine multi-hop and name-collision
cases.

The script is resumable and backs off on HTTP 429. Both matter: the API
rate-limits hard enough that a few dozen articles cannot be fetched in one
burst, and losing completed work to a partial failure would silently shrink
the corpus and therefore the question set.

Usage:
    python tools/fetch_wikipedia.py data/corpus.raw.jsonl
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"

# Wikipedia's policy requires a descriptive User-Agent identifying the tool.
UA = "cfbench-corpus-capture/0.1 (citation-faithfulness-benchmark; research use)"

# Grouped by domain so the question set can be stratified, and chosen for
# articles that carry hard numbers. Within each group the articles overlap
# enough to support multi-hop questions, and several pairs invite
# name-collision and recency traps: two battery plants, two canals, two space
# telescopes, two launch vehicles, companies that all report revenue and
# headcount.
TITLES = [
    # Grid energy storage
    "Grid energy storage",
    "Hornsdale Power Reserve",
    "Moss Landing Power Plant",
    "Pumped-storage hydroelectricity",
    "Lithium iron phosphate battery",
    "Vanadium redox battery",
    "Compressed-air energy storage",
    # Semiconductors
    "TSMC",
    "ASML Holding",
    "Extreme ultraviolet lithography",
    "Moore's law",
    "Wafer (electronics)",
    "GlobalFoundries",
    # Shipping and logistics
    "Maersk",
    "Panama Canal",
    "Suez Canal",
    "Port of Rotterdam",
    "Port of Shanghai",
    "Containerization",
    "Union Pacific Railroad",
    # Spaceflight
    "Falcon 9",
    "James Webb Space Telescope",
    "Hubble Space Telescope",
    "International Space Station",
    "Voyager 1",
    "Ariane 6",
    # Aviation
    "Boeing 787 Dreamliner",
    "Airbus A350",
    "Heathrow Airport",
    "Singapore Changi Airport",
    # Public health
    "Polio vaccine",
    "BCG vaccine",
    "Insulin (medication)",
    "Measles vaccine",
    # Renewables
    "Wind power in Denmark",
    "Solar power in California",
    "Three Gorges Dam",
]

NEWLINE = chr(10)


def fetch_extract(title: str) -> tuple[str, str] | None:
    """Return (canonical_url, plaintext) for an article, or None if missing."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "titles": title,
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)

    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    extract = page.get("extract", "")
    if not extract:
        return None
    return page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"), extract


def fetch_with_backoff(title: str, attempts: int = 6) -> tuple[str, str] | None:
    """Fetch, retrying on HTTP 429 with exponential backoff."""
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            return fetch_extract(title)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts:
                raise
            print(f"    429, backing off {delay:.0f}s ({title})")
            time.sleep(delay)
            delay *= 2
    return None


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + NEWLINE)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/corpus.raw.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_existing(out)
    have = {r["title"] for r in rows}
    if have:
        print(f"resuming: {len(have)} articles already captured")

    failed: list[str] = []
    for i, title in enumerate(TITLES, 1):
        if title in have:
            continue
        try:
            result = fetch_with_backoff(title)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failed.append(f"{title}: {type(exc).__name__}: {exc}")
            continue
        if result is None:
            failed.append(f"{title}: no extract returned")
            continue

        url, raw = result
        rows.append({"url": url, "title": title, "raw": raw})
        print(f"[{i}/{len(TITLES)}] {title}: {len(raw.split())} words")

        # Persist after every article so an interruption costs one fetch.
        save(out, rows)
        time.sleep(2.0)

    save(out, rows)
    total = sum(len(r["raw"].split()) for r in rows)
    print(f"{NEWLINE}{len(rows)} articles total ({total:,} words) in {out}")
    if failed:
        print(f"{len(failed)} failed:", file=sys.stderr)
        for line in failed:
            print(f"  - {line}", file=sys.stderr)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
