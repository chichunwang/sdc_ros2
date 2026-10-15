#!/usr/bin/env python3
"""Inspect track files without assuming pose columns, frame IDs or coordinate systems."""

import argparse
import json
from pathlib import Path
import sys


def read_tracks(config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tracks = config.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise ValueError("Configuration needs a non-empty 'tracks' mapping.")
    required = ("directory", "point_cloud_directory", "map_file", "initial_guess_file")
    for name, settings in tracks.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{name}: settings must be an object.")
        for field in required:
            value = settings.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: '{field}' must be a non-empty relative path.")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name}: '{field}' must stay within its data directory.")
        count = settings.get("expected_frames")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"{name}: 'expected_frames' must be a positive integer.")
    return tracks


def inspect_track(data_root, name, settings, strict_count):
    track_dir = data_root / settings["directory"]
    errors = []
    warnings = []
    report = {
        "track": name,
        "directory": str(track_dir),
        "expected_frames": settings["expected_frames"],
        "point_cloud_count": 0,
        "errors": errors,
        "warnings": warnings,
    }
    if not track_dir.is_dir():
        errors.append(f"Track directory is missing: {track_dir}")
        report["ok"] = False
        return report

    for field in ("map_file", "initial_guess_file"):
        file_path = track_dir / settings[field]
        report[field] = str(file_path)
        if not file_path.is_file():
            errors.append(f"Required file is missing: {file_path}")
        elif file_path.stat().st_size == 0:
            errors.append(f"Required file is empty: {file_path}")

    cloud_dir = track_dir / settings["point_cloud_directory"]
    report["point_cloud_directory"] = str(cloud_dir)
    if not cloud_dir.is_dir():
        errors.append(f"Point cloud directory is missing: {cloud_dir}")
    else:
        clouds = sorted(p for p in cloud_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ply")
        report["point_cloud_count"] = len(clouds)
        empty_count = sum(p.stat().st_size == 0 for p in clouds)
        if not clouds:
            errors.append(f"No PLY files found in: {cloud_dir}")
        if empty_count:
            errors.append(f"Found {empty_count} empty PLY files.")
        if len(clouds) != settings["expected_frames"]:
            message = (
                f"Found {len(clouds)} PLY files; the announcement lists "
                f"{settings['expected_frames']} image frames. Confirm against the official frame IDs."
            )
            (errors if strict_count else warnings).append(message)
        report["filename_examples"] = [p.name for p in clouds[:3]]
    report["ok"] = not errors
    return report


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "tracks.json")
    parser.add_argument("--track", default="all", help="Track key in the config, or 'all'.")
    parser.add_argument("--strict-count", action="store_true", help="Treat count differences as errors.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    try:
        tracks = read_tracks(args.config.expanduser())
    except (OSError, ValueError, AttributeError) as error:
        parser.error(f"Cannot read track configuration: {error}")
    if args.track != "all" and args.track not in tracks:
        parser.error(f"Unknown track '{args.track}'; choose from: {', '.join(tracks)}")

    names = list(tracks) if args.track == "all" else [args.track]
    data_root = args.data_root.expanduser().resolve()
    try:
        reports = [inspect_track(data_root, name, tracks[name], args.strict_count) for name in names]
        result = {
            "check_scope": "File presence, non-empty files and PLY counts only; not pose or geometry validation.",
            "data_root": str(data_root),
            "ok": all(report["ok"] for report in reports),
            "tracks": reports,
        }
        encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except OSError as error:
        print(f"Dataset inspection failed: {error}", file=sys.stderr)
        return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

