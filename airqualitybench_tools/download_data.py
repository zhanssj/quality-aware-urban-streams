#!/usr/bin/env python3
"""Download pinned AirQualityBench files; preserve provenance and local checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "xuxing123/AirQualityBench"
REVISION = "bbdcdd257bc7aed69598d423ca034d03f7ce015e"
META_FILES = ["README.md", "selected_nodes_metadata.csv", "global_quality_stats.csv", "scaler.csv"]
# Published SHA256 shown on Hugging Face's file page at the pinned revision.
KNOWN_SHA256 = {"aq_compact_2025.h5": "bbdc1c689f8abbfd12869a4d0addab4ac310c284664f537d88200778a567d6d5"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, choices=range(2021, 2026), default=list(range(2021, 2026)))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "download_manifest.json"
    try:
        # Branch/tag arguments are resolved ONCE, so files cannot come from mixed commits.
        revision = args.revision
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
            revision = HfApi().dataset_info(REPO, revision=revision).sha
        existing = {}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("revision") != revision or previous.get("repo_id") != REPO:
                raise ValueError("This data directory has a different revision. Use a new --out directory.")
            existing = {item["name"]: item for item in previous.get("files", [])}
        manifest = {
            "repo_id": REPO, "repo_type": "dataset", "revision": revision,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "files": list(existing.values()),
            "note": "SHA256 values are local fingerprints. A remote match is claimed only when explicitly recorded.",
        }
        names = META_FILES + [f"aq_compact_{year}.h5" for year in sorted(set(args.years))]
        for name in names:
            print(f"Downloading/checking {name}", flush=True)
            path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=name, revision=revision, local_dir=args.out))
            fingerprint = sha256(path)
            expected = KNOWN_SHA256.get(name) if revision == REVISION else None
            if expected is not None and fingerprint != expected:
                raise ValueError(f"Published SHA256 mismatch: {name}. Delete this file and retry.")
            existing[name] = {
                "name": name, "size_bytes": path.stat().st_size, "sha256": fingerprint,
                "published_sha256_match": True if expected else None,
                "source_url": f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}",
            }
            manifest["files"] = list(existing.values())
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
        print(f"Done: {args.out.resolve()}\nRevision: {revision}")
        print("The adjacency .pkl file is intentionally not downloaded or executed.")
        return 0
    except Exception as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check network access and disk space; rerun the same command. Completed files are reusable.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
