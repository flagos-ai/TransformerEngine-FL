#!/usr/bin/env python3
"""FlagCICD run-history index (stdlib only).

Maintains a local JSONL index of past group-report outcomes so the evidence
module can decide whether a test group has been stable enough to trust runtime
evidence for selection.

Index format (one JSON object per line):
  {
    "schema_version": "flagcicd.history/v1",
    "recorded_at":    "<ISO-8601>",
    "source_sha":     "<40-hex>",
    "platform":       "<name>",
    "device":         "<name>",
    "group":          "<name>",
    "status":         "<passed|test_failed|...>",
    "evidence_digest": "<sha256-hex|null>",
    "duration_s":     <int>
  }

The file is append-only during a run; compaction trims to the last
HISTORY_WINDOW entries per (platform, device, group) key.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_SCHEMA = "flagcicd.history/v1"
HISTORY_WINDOW = 50  # max entries retained per (platform, device, group)
STABLE_PASS_THRESHOLD = 5  # consecutive passes required before evidence is trusted


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_entry(
    history_path: Path,
    *,
    source_sha: str,
    platform: str,
    device: str,
    group: str,
    status: str,
    evidence_digest: str | None = None,
    duration_s: int = 0,
) -> None:
    """Append one outcome line to the history file (creates file if absent)."""
    entry = {
        "schema_version": HISTORY_SCHEMA,
        "recorded_at": _utcnow(),
        "source_sha": source_sha,
        "platform": platform,
        "device": device,
        "group": group,
        "status": status,
        "evidence_digest": evidence_digest,
        "duration_s": duration_s,
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def load_history(history_path: Path) -> list[dict[str, Any]]:
    """Load all valid history entries from a JSONL file."""
    if not history_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with history_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if obj.get("schema_version") == HISTORY_SCHEMA:
                    entries.append(obj)
            except json.JSONDecodeError:
                # Corrupted line — skip silently; history is advisory.
                pass
    return entries


def query_stability(
    history_path: Path,
    platform: str,
    device: str,
    group: str,
    window: int = STABLE_PASS_THRESHOLD,
) -> dict[str, Any]:
    """Return stability summary for one (platform, device, group) triple.

    Returns a dict with:
      stable        – bool; True when the last `window` runs all passed
      pass_streak   – number of consecutive trailing passes
      total_runs    – total recorded runs for this key
      last_status   – most recent status, or None
    """
    entries = load_history(history_path)
    key_entries = [
        e
        for e in entries
        if e["platform"] == platform and e["device"] == device and e["group"] == group
    ]
    # Most recent first
    key_entries.sort(key=lambda e: e.get("recorded_at", ""), reverse=True)

    streak = 0
    for e in key_entries:
        if e.get("status") == "passed":
            streak += 1
        else:
            break

    return {
        "stable": streak >= window,
        "pass_streak": streak,
        "total_runs": len(key_entries),
        "last_status": key_entries[0].get("status") if key_entries else None,
    }


def compact(history_path: Path) -> int:
    """Trim each key to the last HISTORY_WINDOW entries.  Returns lines removed."""
    if not history_path.exists():
        return 0

    entries = load_history(history_path)
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        key = (e.get("platform", ""), e.get("device", ""), e.get("group", ""))
        buckets[key].append(e)

    kept: list[dict] = []
    removed = 0
    for key, bucket in buckets.items():
        bucket.sort(key=lambda e: e.get("recorded_at", ""))
        if len(bucket) > HISTORY_WINDOW:
            removed += len(bucket) - HISTORY_WINDOW
            bucket = bucket[-HISTORY_WINDOW:]
        kept.extend(bucket)

    kept.sort(key=lambda e: e.get("recorded_at", ""))
    tmp = history_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for e in kept:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
    tmp.replace(history_path)
    return removed


def cmd_append(args: Any) -> int:
    """CLI: append one entry from gate group-report output."""
    import sys

    try:
        report = json.loads(Path(args.report).read_text())
    except Exception as exc:
        print(f"::error::history append: cannot read report: {exc}", file=sys.stderr)
        return 1

    append_entry(
        Path(args.history),
        source_sha=report.get("source_sha", ""),
        platform=report.get("platform", ""),
        device=report.get("device", ""),
        group=report.get("group", ""),
        status=report.get("status", "unknown"),
        evidence_digest=report.get("evidence_digest"),
        duration_s=int(report.get("duration_s", 0)),
    )
    return 0


def cmd_compact(args: Any) -> int:
    removed = compact(Path(args.history))
    print(f"history compact: removed {removed} stale entries")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    ap = sub.add_parser("append", help="Append a group-report outcome")
    ap.add_argument("--history", required=True)
    ap.add_argument("--report", required=True)

    cp = sub.add_parser("compact", help="Trim history to window size")
    cp.add_argument("--history", required=True)

    args = parser.parse_args()
    if args.command == "append":
        return cmd_append(args)
    if args.command == "compact":
        return cmd_compact(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
