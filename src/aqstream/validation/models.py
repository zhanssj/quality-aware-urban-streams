import hashlib
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from aqstream.artifacts import file_sha256, write_json
from aqstream.data import DataContractError
from aqstream.features import YearGrid
from aqstream.replay.calendars import make_calendar
from aqstream.replay.config import SCENARIOS
from aqstream.replay.mixture import weighted_training_samples

VARIANTS = ("persistence", "standard", "masks", "age", "quality_aware", "native_missing")
REGIMES = ("clean", "augmented")
MODEL_NAMES = tuple(f"{regime}/{name}" for regime in REGIMES for name in VARIANTS)


def predictors(common, quality, name):
    if name == "standard":
        return common
    if name == "masks":
        return np.column_stack((common, quality[:, :6]))
    if name == "age":
        return np.column_stack((common, quality[:, 6]))
    if name == "quality_aware":
        return np.column_stack((common, quality))
    if name == "native_missing":
        native = common.copy()
        native[:, :6] = np.where(quality[:, :6] == 1, native[:, :6], np.nan)
        native[:, 6:8] = np.where(quality[:, 7:8] == 0, native[:, 6:8], np.nan)
        native[:, 8:10] = np.where(quality[:, 8:9] == 0, native[:, 8:10], np.nan)
        return native
    raise DataContractError(f"Unknown predictor block: {name}")


def predict(suites, common, quality, persistence):
    outputs, timings = [], {}
    for regime in REGIMES:
        for name in VARIANTS:
            started = perf_counter()
            features = predictors(common, quality, name) if name != "persistence" else None
            transform = perf_counter() - started
            started = perf_counter()
            output = (
                persistence.copy()
                if name == "persistence"
                else suites[regime][name].predict(features)
            )
            timings[f"{regime}/{name}"] = {
                "transform_seconds": transform,
                "inference_seconds": perf_counter() - started,
            }
            if output.shape != persistence.shape or not np.isfinite(output).all():
                raise DataContractError("Model returned incomplete/nonfinite forecasts")
            outputs.append(output)
    return np.column_stack(outputs), timings


def fit_suites(baseline, values, valid, config, output, memory):
    # Only the train grid is accepted; caller cannot pass cached validation to this API.
    steps = baseline.validation_start
    if values.shape != valid.shape or len(values) != steps:
        raise DataContractError("Training arrays must end strictly before validation")
    train_grid = YearGrid(baseline.grid.years[:-1], baseline.grid.lengths[:-1])
    if any(year not in (2021, 2022, 2023) for year in train_grid.years):
        raise DataContractError("No validation/test fit")
    for name in ("train_inputs.npz", "training_sample_ids.csv"):
        if file_sha256(baseline.directory / name) != baseline.manifest["artifacts"][name]["sha256"]:
            raise DataContractError(f"Changed trusted training artifact: {name}")
    with np.load(baseline.directory / "train_inputs.npz", allow_pickle=False) as data:
        common, quality, labels = data["common"], data["quality"], data["labels"]
    stride = config["train_stride"]
    original_stride = baseline.manifest["train_scheduled"] // len(baseline.station_ids)
    if original_stride != len(range(0, steps, stride)):
        raise DataContractError("T06 must retain the B01 training target selection")
    expected = values[::stride][valid[::stride]]
    if (
        not np.array_equal(expected, labels)
        or common.shape != (len(labels), 10)
        or quality.shape != (len(labels), 11)
    ):
        raise DataContractError("Trusted B01 train labels/grid differ")
    clean_weight = config["training_mixture_weights"]["clean_source"]
    disturbed = {k: w for k, w in config["training_mixture_weights"].items() if k != "clean_source"}
    seeds = config["train_mixture_seeds"]
    views = 1 + len(disturbed) * len(seeds)
    n = len(labels)
    augmented_common = np.empty((views * n, 10))
    augmented_quality = np.empty((views * n, 11))
    augmented_labels = np.tile(labels, views)
    weights = np.empty(views * n)
    augmented_common[:n], augmented_quality[:n], weights[:n] = common, quality, clean_weight
    position = n
    view_records = [
        {
            "scenario": "clean_source",
            "weight": clean_weight,
            "samples": n,
            "provenance": "exact frozen B01 train matrices",
        }
    ]
    for seed in seeds:
        before = position
        hashes = {}
        for sample in weighted_training_samples(
            values,
            valid,
            baseline.fitted,
            train_grid,
            specs=tuple(SCENARIOS[name] for name in disturbed),
            weights=tuple(w / (1 - clean_weight) for w in disturbed.values()),
            train_seed=seed,
        ):
            target = sample.features.sample_ids[0].target_index
            hashes.setdefault(
                sample.calendar_sha256, (sample.weight * (1 - clean_weight) / len(seeds), 0)
            )
            if target % stride:
                continue
            selected = sample.truth.source_valid
            count = int(selected.sum())
            augmented_common[position : position + count] = sample.features.common[selected]
            augmented_quality[position : position + count] = sample.features.quality[selected]
            weight, previous = hashes[sample.calendar_sha256]
            weights[position : position + count] = weight
            if not np.array_equal(
                sample.truth.values[selected], augmented_labels[position : position + count]
            ):
                raise DataContractError("Augmented view changed labels or their order")
            hashes[sample.calendar_sha256] = (weight, previous + count)
            position += count
        if position - before != n * len(disturbed) or any(
            count != n for _, count in hashes.values()
        ):
            raise DataContractError("Incomplete augmented training view")
        for name, (key, (weight, count)) in zip(disturbed, hashes.items(), strict=True):
            calendar = make_calendar(
                valid[:-1], baseline.station_ids, input_start=0, spec=SCENARIOS[name], seed=seed
            )
            if calendar.fingerprint != key:
                raise DataContractError("Training calendar provenance differs from generated view")
            calendar.save(output / "train_calendars" / calendar.scenario_id)
            view_records.append(
                {
                    "scenario": name,
                    "seed": seed,
                    "calendar_sha256": key,
                    "weight": weight,
                    "samples": count,
                    "statistics": calendar.statistics,
                }
            )
        print(f"T06 train features: seed {seed}, {position}/{views * n} rows", flush=True)
        memory.check()
    if position != len(weights) or not np.isclose(weights.sum(), n):
        raise DataContractError("Augmented samples must retain total original training weight")
    fit_plan = {
        "training_years": train_grid.years,
        "target_stride": stride,
        "original_fit_labels": n,
        "augmented_rows": len(weights),
        "total_weight": float(weights.sum()),
        "views": view_records,
        "common_sha256": hashlib.sha256(augmented_common.tobytes()).hexdigest(),
        "quality_sha256": hashlib.sha256(augmented_quality.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(augmented_labels.tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "original_sample_ids_sha256": file_sha256(baseline.directory / "training_sample_ids.csv"),
        "model_params": baseline.models.standard.get_params(),
        "validation_numeric_read_before_fit": False,
    }
    write_json(output / "fit_plan.json", fit_plan)
    suites = {
        "clean": {
            "standard": baseline.models.standard,
            "quality_aware": baseline.models.quality_aware,
        },
        "augmented": {},
    }
    fit_times = {}
    for regime, x, q, y, w in (
        ("clean", common, quality, labels, None),
        ("augmented", augmented_common, augmented_quality, augmented_labels, weights),
    ):
        for name in VARIANTS[1:]:
            if name in suites[regime]:
                continue
            estimator = HistGradientBoostingRegressor(**baseline.models.standard.get_params())
            started = perf_counter()
            estimator.fit(predictors(x, q, name), y, sample_weight=w)
            fit_times[f"{regime}/{name}"] = perf_counter() - started
            suites[regime][name] = estimator
            memory.check()
            print(f"T06 fit {regime}/{name}: {fit_times[f'{regime}/{name}']:.3f}s", flush=True)
    write_json(output / "fit_timings.json", fit_times)
    return suites, fit_plan
