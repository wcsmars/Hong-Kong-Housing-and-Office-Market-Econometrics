#!/usr/bin/env python3
"""Check external input presence and snapshot hashes without network access.

Run from any working directory. Only the Python standard library is required.
Updated inputs are allowed by default; --strict-snapshot requires exact hashes.
This check does not validate file schemas or reproduce the research analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def safe_relative_path(value: str) -> Path:
    """Reject absolute and parent-relative locations in the manifest."""
    if not isinstance(value, str) or not value:
        raise ValueError("manifest paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"invalid manifest path: {value!r}")
    return Path(*path.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("housing", "office", "all"), default="all")
    parser.add_argument(
        "--strict-snapshot", action="store_true",
        help="fail if any present file differs from its recorded SHA256 snapshot",
    )
    args = parser.parse_args()
    manifest_path = PUBLIC_ROOT / "input_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["manifest_version"] != 1:
            raise ValueError("unsupported manifest version")
        studies = manifest["studies"]
        selected = ("housing", "office") if args.study == "all" else (args.study,)
        checks = []
        for study in selected:
            config = studies[study]
            directory = safe_relative_path(config["directory"])
            entries = config["inputs"]
            if config["required_input_count"] != len(entries):
                raise ValueError(f"{study}: manifest input count does not match entries")
            files = []
            seen = set()
            for entry in entries:
                relative = directory / safe_relative_path(entry["path"])
                if relative in seen:
                    raise ValueError(f"{study}: duplicate input {entry['path']}")
                seen.add(relative)
                expected = entry["sha256"]
                if not isinstance(expected, str) or len(expected) != 64 or any(
                    c not in "0123456789abcdef" for c in expected
                ):
                    raise ValueError(f"{study}: invalid SHA256 for {entry['path']}")
                files.append((relative, expected))
            checks.append((study, files))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"cannot load input_manifest.json: {error}")

    missing_count = changed_count = unreadable_count = 0
    for study, files in checks:
        missing, changed, unreadable = [], [], []
        for relative, expected in files:
            path = PUBLIC_ROOT / relative
            if not path.is_file():
                missing.append(relative)
                continue
            try:
                actual = sha256(path)
            except OSError as error:
                unreadable.append((relative, error.strerror or type(error).__name__))
                continue
            if actual != expected:
                changed.append(relative)

        present = len(files) - len(missing)
        print(f"{study}: {present}/{len(files)} required files present; "
              f"{len(changed)} snapshot differences")
        if missing:
            print("  Missing:")
            for relative in missing:
                print(f"    {relative.as_posix()}")
        if changed:
            level = "ERROR" if args.strict_snapshot else "WARNING"
            print(f"  {level}: SHA256 differs from the recorded snapshot:")
            for relative in changed:
                print(f"    {relative.as_posix()}")
        if unreadable:
            print("  Unreadable:")
            for relative, reason in unreadable:
                print(f"    {relative.as_posix()}: {reason}")
        missing_count += len(missing)
        changed_count += len(changed)
        unreadable_count += len(unreadable)

    if missing_count or unreadable_count:
        print("Input check failed. See input_manifest.json for source and schema notes.")
        return 1
    if args.strict_snapshot and changed_count:
        print("Input check failed: strict snapshot matching is enabled.")
        return 1
    if changed_count:
        print("Input check passed with snapshot warnings; updated data may change results.")
    else:
        print("Input check passed: all required files match the recorded snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
