#!/usr/bin/env python3
"""Artifact identity management: write manifest during build, verify during install."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import flagcicd_contract as contract


def cmd_write(args: argparse.Namespace) -> int:
    """Write artifact identity manifest during build."""
    # Collect wheel metadata
    wheels = []
    artifact_dir = Path(args.artifact_dir)
    if artifact_dir.exists():
        for wheel_path in sorted(artifact_dir.glob("*.whl")):
            wheel_bytes = wheel_path.read_bytes()
            wheels.append(
                {
                    "name": wheel_path.name,
                    "sha256": hashlib.sha256(wheel_bytes).hexdigest(),
                    "size": len(wheel_bytes),
                }
            )

    # Get toolchain info
    toolchain = {}
    try:
        import platform as plat
        import subprocess

        toolchain["platform"] = plat.platform()
        toolchain["python_version"] = plat.python_version()

        # Try to get torch version
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            toolchain["torch_version"] = result.stdout.strip()
    except Exception:
        pass

    # Build identity
    identity = {
        "schema_version": contract.IDENTITY_SCHEMA,
        "platform": args.platform,
        "source_sha": args.source_sha,
        "config_hash": args.config_hash,
        "image_ref": args.image_ref,
        "image_digest": args.image_digest or "",
        "test_level": args.test_level,
        "coverage_instrumentation": args.coverage_active,
        "toolchain": toolchain,
        "wheels": wheels,
        "created_at": args.timestamp or "",
    }

    # Write identity file
    identity_path = artifact_dir / "artifact-identity.json"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    print(f"Wrote artifact identity: {identity_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify artifact identity matches plan."""
    # Load identity
    identity_path = Path(args.artifact_dir) / "artifact-identity.json"
    if not identity_path.exists():
        print(f"ERROR: Identity file not found: {identity_path}", file=sys.stderr)
        return 1

    try:
        identity = json.loads(identity_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Failed to load identity: {exc}", file=sys.stderr)
        return 1

    if not isinstance(identity, dict):
        print("ERROR: Identity must be an object", file=sys.stderr)
        return 1

    # Verify both the declared identity and the downloaded payload.
    errors = []
    if identity.get("schema_version") != contract.IDENTITY_SCHEMA:
        errors.append("Unsupported artifact identity schema")
    wheels = identity.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        errors.append("Identity must contain at least one wheel")
        wheels = []
    names = set()
    artifact_dir = Path(args.artifact_dir)
    for wheel in wheels:
        if not isinstance(wheel, dict):
            errors.append("Invalid wheel entry")
            continue
        name = wheel.get("name")
        if (
            not isinstance(name, str)
            or not name.endswith(".whl")
            or Path(name).name != name
            or "\\" in name
            or name in names
        ):
            errors.append(f"Invalid or duplicate wheel name: {name!r}")
            continue
        names.add(name)
        try:
            payload = (artifact_dir / name).read_bytes()
            if type(wheel.get("size")) is not int or len(payload) != wheel["size"]:
                errors.append(f"Wheel size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != wheel.get("sha256"):
                errors.append(f"Wheel SHA256 mismatch: {name}")
        except OSError as exc:
            errors.append(f"Cannot read wheel {name}: {exc}")
    if names != {path.name for path in artifact_dir.glob("*.whl")}:
        errors.append("Wheel files do not match identity manifest")
    if identity.get("platform") != args.platform:
        errors.append(
            f"Platform mismatch: expected {args.platform}, got {identity.get('platform')}"
        )

    if identity.get("source_sha") != args.source_sha:
        errors.append(
            f"Source SHA mismatch: expected {args.source_sha[:12]}, got"
            f" {identity.get('source_sha', '')[:12]}"
        )

    if identity.get("config_hash") != args.config_hash:
        errors.append(
            f"Config hash mismatch: expected {args.config_hash[:12]}, got"
            f" {identity.get('config_hash', '')[:12]}"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Artifact identity verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # write
    write_parser = subparsers.add_parser("write", help="Write identity manifest")
    write_parser.add_argument("--platform", required=True)
    write_parser.add_argument("--source-sha", required=True)
    write_parser.add_argument("--config-hash", required=True)
    write_parser.add_argument("--image-ref", required=True)
    write_parser.add_argument("--image-digest", default="")
    write_parser.add_argument("--test-level", required=True)
    write_parser.add_argument("--coverage-active", action="store_true")
    write_parser.add_argument("--artifact-dir", type=Path, required=True)
    write_parser.add_argument("--timestamp", default="")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify identity")
    verify_parser.add_argument("--platform", required=True)
    verify_parser.add_argument("--source-sha", required=True)
    verify_parser.add_argument("--config-hash", required=True)
    verify_parser.add_argument("--artifact-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "write":
        return cmd_write(args)
    elif args.command == "verify":
        return cmd_verify(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
