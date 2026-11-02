#!/usr/bin/env python3
"""Read-only structural and numerical HDF5 audit. No model fitting or imputation."""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, bytes):
        return text(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_param(value: str) -> str:
    value = value.lower().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return re.sub(r"[^a-z0-9]", "", value)


def audit_one(path: Path, block_rows: int = 168) -> tuple[dict, list[dict], list[dict]]:
    year = int(re.fullmatch(r"aq_compact_(\d{4})\.h5", path.name).group(1))
    warnings: list[str] = []
    errors: list[str] = []
    result: dict = {
        "file": path.name, "year_from_filename": year,
        "size_bytes": path.stat().st_size, "sha256": sha256(path),
        "warnings": warnings, "errors": errors,
        "time_axis_validated": False,
        "time_note": "Row counts and filename year do not prove timestamps, timezone or hourly alignment. Gap lengths below are row steps, not verified hours.",
    }
    with h5py.File(path, "r") as f:
        inventory: dict[str, dict] = {}
        def inspect(name: str, obj: Any) -> None:
            # Do not read external links or execute serialized objects.
            if isinstance(obj, h5py.Dataset):
                inventory[name] = {"shape": obj.shape, "dtype": str(obj.dtype), "chunks": obj.chunks,
                                   "compression": obj.compression, "attrs": {k: jsonable(v) for k, v in obj.attrs.items()}}
        f.visititems(inspect)
        result["datasets"] = inventory
        result["file_attrs"] = {k: jsonable(v) for k, v in f.attrs.items()}
        result["temporal_dataset_candidates"] = [name for name in inventory if any(token in name.lower() for token in ("time", "date"))]
        for key in ("values", "masks", "station_ids", "params"):
            if key not in f or not isinstance(f[key], h5py.Dataset):
                raise ValueError(f"{path.name}: required dataset {key!r} is missing")
        values, masks = f["values"], f["masks"]
        if values.ndim != 3 or values.shape != masks.shape:
            raise ValueError(f"{path.name}: incompatible values/masks shapes: {values.shape}, {masks.shape}")
        if values.dtype.kind not in "fiu" or masks.dtype.kind not in "bifu":
            raise ValueError(f"{path.name}: unsupported values/masks dtype")
        t, n, p = values.shape
        if min(t, n, p) == 0:
            raise ValueError(f"{path.name}: empty tensor")
        if f["station_ids"].ndim != 1 or f["params"].ndim != 1:
            raise ValueError(f"{path.name}: station_ids and params must be one-dimensional")
        ids = [text(v) for v in f["station_ids"][:]]
        params = [text(v) for v in f["params"][:]]
        if len(ids) != n or len(params) != p:
            raise ValueError(f"{path.name}: axis lengths do not match station_ids/params")
        if len(set(ids)) != n:
            errors.append("Duplicate station_ids")
        if len(set(params)) != p:
            errors.append("Duplicate pollutant names")
        expected_rows = 8784 if calendar.isleap(year) else 8760
        if t != expected_rows:
            warnings.append(f"{t} rows, not the expected {expected_rows} for a complete hourly calendar year")
        if (n, p) != (3720, 6):
            warnings.append(f"Station/parameter counts {(n, p)} differ from published (3720, 6)")
        pm_indices = [i for i, name in enumerate(params) if normalize_param(name) in ("pm25", "pm2p5")]
        if len(pm_indices) != 1:
            raise ValueError(f"{path.name}: cannot uniquely identify PM2.5 among {params}")
        pm = pm_indices[0]
        shape = (n, p)
        names = ["mask_true", "usable", "mask_true_nonfinite", "negative", "zero", "invalid_mask", "masked_finite", "masked_zero"]
        counters = {name: np.zeros(shape, dtype=np.int64) for name in names}
        sums = np.zeros(p, dtype=np.float64)
        sums_sq = np.zeros(p, dtype=np.float64)
        minima = np.full(p, np.inf)
        maxima = np.full(p, -np.inf)
        current_gap = np.zeros(n, dtype=np.int64)
        max_gap = np.zeros(n, dtype=np.int64)
        max_internal_gap = np.zeros(n, dtype=np.int64)
        first = np.full(n, -1, dtype=np.int64)
        last = np.full(n, -1, dtype=np.int64)
        for start in range(0, t, block_rows):
            end = min(t, start + block_rows)
            v = values[start:end, :, :]
            m = masks[start:end, :, :]
            finite = np.isfinite(v)
            observed = m == 1
            missing = m == 0
            good = observed & finite
            flags = {
                "mask_true": observed, "usable": good,
                "mask_true_nonfinite": observed & ~finite,
                "negative": good & (v < 0), "zero": good & (v == 0),
                "invalid_mask": ~(observed | missing),
                "masked_finite": missing & finite, "masked_zero": missing & finite & (v == 0),
            }
            for name, flag in flags.items():
                counters[name] += flag.sum(axis=0, dtype=np.int64)
            for j in range(p):
                x = v[:, :, j][good[:, :, j]].astype(np.float64)
                if x.size:
                    sums[j] += x.sum()
                    sums_sq[j] += np.dot(x, x)
                    minima[j] = min(minima[j], float(x.min()))
                    maxima[j] = max(maxima[j], float(x.max()))
            for offset, row in enumerate(good[:, :, pm]):
                idx = start + offset
                returning = row & (first >= 0)
                max_internal_gap = np.where(returning, np.maximum(max_internal_gap, current_gap), max_internal_gap)
                first = np.where(row & (first < 0), idx, first)
                last = np.where(row, idx, last)
                current_gap = np.where(row, 0, current_gap + 1)
                max_gap = np.maximum(max_gap, current_gap)
        pollutant_rows = []
        for j, name in enumerate(params):
            counts = {key: int(value[:, j].sum()) for key, value in counters.items()}
            usable = counts["usable"]
            mean = float(sums[j] / usable) if usable else None
            std = float(np.sqrt(max(0.0, sums_sq[j] / usable - mean * mean))) if usable else None
            pollutant_rows.append({
                "year": year, "pollutant": name, "total_cells": t * n,
                **counts, "usable_fraction": usable / (t * n),
                "min_observed_finite": float(minima[j]) if usable else None,
                "max_observed_finite": float(maxima[j]) if usable else None,
                "mean_observed_finite": mean, "std_observed_finite": std,
            })
        for bad in ("mask_true_nonfinite", "invalid_mask"):
            count = int(counters[bad].sum())
            if count:
                errors.append(f"{bad}: {count}")
        negative_total = int(counters["negative"].sum())
        if negative_total:
            warnings.append(f"Negative observed finite values: {negative_total}; flag for review, do not silently delete")
        counts = counters["usable"][:, pm]
        coverage = counts / t
        result.update({
            "shape": [t, n, p], "station_ids": ids, "params": params,
            "pm25_param_index": pm, "pm25_usable_cells": int(counts.sum()),
            "pm25_usable_fraction": float(coverage.mean()),
            "pm25_stations_no_usable": int((counts == 0).sum()),
            "pm25_stations_coverage_ge_80pct": int((coverage >= .8).sum()),
            "pm25_stations_coverage_ge_90pct": int((coverage >= .9).sum()),
            "pm25_stations_coverage_ge_95pct": int((coverage >= .95).sum()),
            "pollutants": pollutant_rows,
        })
        warnings.append("Physical units, source calibration and actual arrival times are not validated by this numerical audit")
        station_rows = []
        for i, sid in enumerate(ids):
            station_rows.append({
                "year": year, "station_id": sid, "rows": t,
                "pm25_mask_true": int(counters["mask_true"][i, pm]),
                "pm25_usable": int(counts[i]), "pm25_usable_fraction": float(coverage[i]),
                "pm25_unusable_fraction": float(1.0 - coverage[i]),
                "pm25_negative": int(counters["negative"][i, pm]),
                "pm25_zero": int(counters["zero"][i, pm]),
                "pm25_mask_true_nonfinite": int(counters["mask_true_nonfinite"][i, pm]),
                "first_usable_row": int(first[i]) if first[i] >= 0 else None,
                "last_usable_row": int(last[i]) if last[i] >= 0 else None,
                "longest_unusable_run_steps": int(max_gap[i]),
                "longest_internal_unusable_run_steps": int(max_internal_gap[i]),
                "leading_unusable_steps": int(first[i]) if first[i] >= 0 else t,
                "trailing_unusable_steps": t - 1 - int(last[i]) if last[i] >= 0 else t,
            })
        return result, pollutant_rows, station_rows


def metadata_check(path: Path, file_reports: list[dict]) -> dict:
    if not path.exists():
        return {"present": False, "note": "No coordinate audit"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"station_id", "lat", "lon"}
    if not rows or not required.issubset(rows[0]):
        return {"present": True, "error": "Metadata is empty or lacks station_id/lat/lon"}
    ids = [row["station_id"].strip() for row in rows]
    invalid = []
    for row in rows:
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
            ok = np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
        except (ValueError, TypeError):
            ok = False
        if not ok:
            invalid.append(row["station_id"])
    ids_set = set(ids)
    return {"present": True, "row_count": len(rows), "duplicate_id_count": len(ids) - len(ids_set),
            "invalid_coordinate_station_ids": invalid,
            "h5_ids_missing_from_metadata": {r["file"]: sorted(set(r["station_ids"]) - ids_set) for r in file_reports}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("audit"))
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--block-rows", type=int, default=168)
    args = parser.parse_args()
    if args.block_rows < 1:
        parser.error("--block-rows must be positive")
    if args.years:
        files = [args.data / f"aq_compact_{y}.h5" for y in sorted(set(args.years))]
        absent = [str(f) for f in files if not f.is_file()]
        if absent:
            parser.error("Files not found: " + ", ".join(absent))
    else:
        files = sorted(f for f in args.data.glob("aq_compact_*.h5") if re.fullmatch(r"aq_compact_\d{4}\.h5", f.name))
    if not files:
        parser.error("No aq_compact_YYYY.h5 files found. Download the dataset first.")
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "Full values/masks scan for the listed local HDF5 files; not verification against original providers",
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("h5py", "numpy")},
        "files": [], "failures": [], "cross_year_checks": [],
        "limitations": [
            "No confirmation of timestamp semantics, timezone, units or calibration",
            "A missing cell does not identify its cause: outage, absent sensor, filtering or other causes",
            "Long runs include station inactivity; internal gaps are reported separately",
            "No reconstruction of true arrival timestamps",
            "Do not choose stations, tune models, or fit scalers using 2025 test outcomes",
            "scaler.csv is not applied; fit any scaler on training data only",
        ],
    }
    manifest_path = args.data / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    summary["download_manifest"] = manifest
    expected = {r["name"]: r for r in (manifest or {}).get("files", [])}
    pollutant_rows, station_rows = [], []
    for path in files:
        print(f"Auditing {path.name} (full values/masks scan)...", flush=True)
        try:
            result, pollutants, stations = audit_one(path, args.block_rows)
            record = expected.get(path.name)
            result["matches_download_manifest"] = result["sha256"] == record["sha256"] if record else None
            if result["matches_download_manifest"] is False:
                result["errors"].append("Local SHA256 differs from download manifest")
            summary["files"].append(result)
            pollutant_rows.extend(pollutants)
            station_rows.extend(stations)
        except Exception as exc:
            summary["failures"].append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    good_files = summary["files"]
    if good_files:
        base = good_files[0]
        for other in good_files[1:]:
            summary["cross_year_checks"].append({
                "reference": base["file"], "other": other["file"],
                "station_ids_same_set": set(base["station_ids"]) == set(other["station_ids"]),
                "station_ids_same_order": base["station_ids"] == other["station_ids"],
                "params_same_order": base["params"] == other["params"],
            })
    try:
        summary["metadata"] = metadata_check(args.data / "selected_nodes_metadata.csv", good_files)
    except Exception as exc:
        summary["metadata"] = {"error": f"{type(exc).__name__}: {exc}"}
    csv_write(args.out / "pollutant_year_summary.csv", pollutant_rows)
    csv_write(args.out / "pm25_station_year.csv", station_rows)
    paths = ["summary.json", "report.md"]
    if pollutant_rows:
        paths.append("pollutant_year_summary.csv")
    if station_rows:
        paths.append("pm25_station_year.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# AirQualityBench: локальный аудит", "", f"Проверено файлов: {len(good_files)}. Ошибок открытия/сканирования: {len(summary['failures'])}.", "",
             "Показатели ниже рассчитаны по локальным файлам. Годы взяты из имён файлов. Календарная ось, единицы и достоверность измерений независимо не подтверждены.", "",
             "Доступное значение (usable): mask == 1 и конечное число. Отрицательные значения отмечаются отдельно, но автоматически не удаляются.", "",
             "| Год | Строки | Станции | Доступность PM2.5 | Станции ≥80% | Без PM2.5 |", "|---|---:|---:|---:|---:|---:|"]
    for r in good_files:
        lines.append(f"| {r['year_from_filename']} | {r['shape'][0]} | {r['shape'][1]} | {r['pm25_usable_fraction']:.2%} | {r['pm25_stations_coverage_ge_80pct']} | {r['pm25_stations_no_usable']} |")
    lines.extend(["", "## Проблемы и оговорки"])
    for r in good_files:
        lines.extend([f"", f"### {r['file']}"] + [f"- ERROR: {w}" for w in r["errors"]] + [f"- WARNING: {w}" for w in r["warnings"]])
    for failure in summary["failures"]:
        lines.append(f"- {failure['file']}: {failure['error']}")
    lines.extend(["", "## Ограничения", *[f"- {s}" for s in summary["limitations"]], "",
                  "Проверки порядка станций между годами, перечень HDF5-полей и проверка координат находятся в summary.json.",
                  "Ни одна подвыборка станций не отбирается автоматически. Статистика тестового периода не должна управлять подбором модели."])
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    zip_path = args.out / "audit_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            archive.write(args.out / name, arcname=name)
    print(f"Results: {zip_path.resolve()}")
    return 1 if summary["failures"] or any(r["errors"] for r in good_files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
