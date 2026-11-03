import hashlib
import json
import re
import zipfile
from pathlib import Path

from aqstream.data import DataContractError


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_review(root: Path, output: Path) -> Path:
    """Whitelisted small source/docs/evidence only; never follow filesystem links."""
    root = root.resolve()
    patterns = [
        "src/aqstream/**/*.py",
        "tests/*.py",
        "docs/*.md",
        "tasks/*.md",
        "handoffs/B01.md",
        "handoffs/T02.md",
        "handoffs/T03.md",
        "handoffs/B02.md",
        "handoffs/T04.md",
        "handoffs/T05.md",
        "configs/*.yaml",
        ".github/workflows/*.yml",
    ]
    fixed = [
        "AGENTS.md",
        "PROJECT.md",
        "STATUS.md",
        "README.md",
        "IMPLEMENTATION_PLAN.md",
        "pyproject.toml",
        "requirements.lock.txt",
        ".gitignore",
        ".gitattributes",
        "manifests/cohort_candidates_669.csv",
        "reviews/T01-R1.md",
        "reviews/B01.md",
        "reviews/T01.md",
        "context/previous_audit_review_ru.md",
        "manifests/context_files.json",
        "prompts/CODEX_B01_PROMPT.txt",
        ".env.example",
        "reports/B01/validation_metrics.json",
        "reports/B01/pilot_summary.json",
        "reports/B01/verification.json",
        "reports/B01/cohort.json",
        "reports/B01/self_review.md",
        "reports/B01/pytest.txt",
        "reports/B01/T02_gate.txt",
        "reports/B01/ruff_check.txt",
        "reports/B01/ruff_format.txt",
        "reports/B01/synthetic_e2e.txt",
        "reports/B01/real_pilot.txt",
        "reports/B02/README.md",
        "reports/B02/pilot_summary.json",
        "reports/B02/pilot_metrics.json",
        "reports/B02/smoke_summary.json",
        "reports/B02/zero_control.json",
        "reports/B02/verification.json",
        "reports/B02/self_review.md",
        "reports/B02/metrics_table.csv",
        "reports/B02/seeds_summary.csv",
        "reports/B02/pytest.txt",
        "reports/B02/N1_gate.txt",
        "reports/B02/T04_gate.txt",
        "reports/B02/T05_gate.txt",
        "reports/B02/synthetic_e2e.txt",
        "reports/B02/ruff_check.txt",
        "reports/B02/ruff_format.txt",
        "reports/B02/smoke.txt",
        "reports/B02/pilot.txt",
    ]
    candidates = {root / name for name in fixed if (root / name).is_file()}
    for pattern in patterns:
        candidates.update(root.glob(pattern))
    payloads = {}
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if any(
            part.is_symlink() or part.is_junction()
            for part in (path, *path.parents)
            if part != root
        ):
            raise DataContractError(f"Review export refuses linked path: {path}")
        if not path.resolve().is_relative_to(root) or path.stat().st_size > 2 * 1024 * 1024:
            raise DataContractError(f"Review export refuses external/large file: {path}")
        data = path.read_bytes()
        if re.search(
            rb"gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
            rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY",
            data,
        ):
            raise DataContractError(f"Possible secret in review file: {path}")
        payloads[path.relative_to(root).as_posix()] = data
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr(
            "bundle_sha256.json",
            json.dumps(
                {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
                indent=2,
            ),
        )
    return output
