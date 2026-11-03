import hashlib
from pathlib import Path

import numpy as np

from aqstream.data import DataContractError
from aqstream.prediction_plan import PredictionPlan
from aqstream.validation.models import MODEL_NAMES, REGIMES, VARIANTS

SLICE_NAMES = ("ordinary", "disruption", "recovery")


def slice_labels(calendar, wait, recovery_steps):
    disrupted = calendar.dropped | (calendar.delay_minutes > wait)
    labels = disrupted.astype(np.uint8)
    last = np.full(disrupted.shape[1], -recovery_steps - 1)
    for step, row in enumerate(disrupted):
        labels[step, ~row & (step - last <= recovery_steps)] = 2
        last = np.where(row, step, last)
    return labels


def block_bootstrap(delta, valid, *, block_steps, replicates, seed):
    """Resample time blocks; every station inside the sampled block stays together."""
    if delta.shape != valid.shape or delta.ndim != 2 or valid.dtype != bool:
        raise DataContractError("Bootstrap needs aligned time x station arrays")
    if block_steps < 1 or replicates < 1 or not np.isfinite(delta[valid]).all():
        raise DataContractError("Invalid bootstrap inputs")
    starts = np.arange(0, len(delta), block_steps)
    sums = np.add.reduceat(np.where(valid, delta, 0).sum(axis=1), starts)
    counts = np.add.reduceat(valid.sum(axis=1), starts)
    if not counts.sum():
        return {
            "delta_mae": None,
            "ci95": [None, None],
            "blocks": len(starts),
            "valid_replicates": 0,
        }
    choices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(starts), (replicates, len(starts))
    )
    numerator, denominator = sums[choices].sum(axis=1), counts[choices].sum(axis=1)
    samples = numerator[denominator > 0] / denominator[denominator > 0]
    return {
        "delta_mae": float(sums.sum() / counts.sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "blocks": len(starts),
        "valid_replicates": len(samples),
        "block_steps": block_steps,
        "replicates": replicates,
        "seed": seed,
        "stations_resampled_independently": False,
    }


def score_arrays(predictions, truth, valid, plan, labels, *, bootstrap):
    shape = (plan.target_stop - plan.target_start, len(plan.station_ids))
    if (
        predictions.shape != (*shape, len(MODEL_NAMES))
        or truth.shape != shape
        or valid.shape != shape
        or valid.dtype != bool
    ):
        raise DataContractError("Tensor does not cover the expected Cartesian grid")
    if (
        labels.shape != shape
        or not np.isin(labels, [0, 1, 2]).all()
        or not np.isfinite(truth[valid]).all()
    ):
        raise DataContractError("Invalid truth or evaluation slices")
    finite = np.isfinite(predictions)
    scored = valid & finite.all(axis=2)
    digest = hashlib.sha256()
    for row, column in np.argwhere(scored):
        target = plan.target_start + int(row)
        digest.update(f"{plan.station_ids[column]}:{target - 1}:{target}\n".encode())
    errors = np.where(scored[:, :, None], predictions - truth[:, :, None], 0)
    if not np.isfinite(errors).all():
        raise DataContractError("Nonfinite error arithmetic")
    absolute = np.abs(errors)
    models = {}
    for index, name in enumerate(MODEL_NAMES):
        per_station = []
        counts = scored.sum(axis=0)
        for column, station in enumerate(plan.station_ids):
            n = int(counts[column])
            per_station.append(
                {
                    "station_id": station,
                    "scored": n,
                    "mae": float(absolute[:, column, index].sum() / n) if n else None,
                }
            )
        n = int(scored.sum())
        square = np.square(errors[:, :, index]).sum()
        if not np.isfinite(square):
            raise DataContractError("Nonfinite squared error arithmetic")
        details = {}
        for code, slice_name in enumerate(SLICE_NAMES):
            selected = scored & (labels == code)
            count = int(selected.sum())
            details[slice_name] = {
                "scheduled": int((labels == code).sum()),
                "scored": count,
                "mae": float(absolute[:, :, index][selected].mean()) if count else None,
                "rmse": float(np.sqrt(np.square(errors[:, :, index][selected]).mean()))
                if count
                else None,
            }
        station_mae = [item["mae"] for item in per_station if item["scored"]]
        models[name] = {
            "mae": float(absolute[:, :, index].sum() / n) if n else None,
            "rmse": float(np.sqrt(square / n)) if n else None,
            "macro_station_mae": float(np.mean(station_mae)) if station_mae else None,
            "scheduled": plan.expected_count,
            "predicted": int(finite[:, :, index].sum()),
            "prediction_coverage": float(finite[:, :, index].mean()),
            "scored": n,
            "eligible_truth": int(valid.sum()),
            "skipped_missing_truth": int((~valid).sum()),
            "skipped_missing_any_prediction": int(valid.sum() - n),
            "per_station": per_station,
            "slices": details,
        }
    paired = {}
    comparisons = []
    for regime in REGIMES:
        comparisons.extend(
            (f"{regime}/{left}", f"{regime}/{right}")
            for left, right in (
                ("quality_aware", "standard"),
                ("quality_aware", "persistence"),
                ("masks", "standard"),
                ("age", "standard"),
                ("native_missing", "standard"),
            )
        )
    comparisons.extend((f"augmented/{name}", f"clean/{name}") for name in VARIANTS)
    for left, right in comparisons:
        delta = absolute[:, :, MODEL_NAMES.index(left)] - absolute[:, :, MODEL_NAMES.index(right)]
        paired[f"{left} minus {right}"] = block_bootstrap(delta, scored, **bootstrap)
    return {
        "schedule_validated": True,
        "planned": plan.expected_count,
        "present_rows": plan.expected_count,
        "expected_plan_sha256": plan.fingerprint,
        "context": {
            "run_id": plan.run_id,
            "scenario_id": plan.scenario_id,
            "wait_minutes": plan.wait_minutes,
        },
        "scored_target_ids_sha256": digest.hexdigest(),
        "units_verified": False,
        "units": "file_units",
        "interpretation": "diagnostic_only",
        "models": models,
        "paired_mae": paired,
    }


def save_predictions(path, predictions, plan, labels):
    np.savez_compressed(
        path,
        predictions=predictions,
        slices=labels,
        station_ids=np.array(plan.station_ids),
        target_indices=np.arange(plan.target_start, plan.target_stop),
        model_names=np.array(MODEL_NAMES),
        plan_sha256=np.array(plan.fingerprint),
        run_id=np.array(plan.run_id),
        scenario_id=np.array(plan.scenario_id),
        wait_minutes=np.array(plan.wait_minutes),
    )


def evaluate_npz(path: Path, truth_path: Path, *, bootstrap, plan=None):
    plan = plan or PredictionPlan.load(path.parent / "evaluation_plan.json")
    try:
        with (
            np.load(path, allow_pickle=False) as data,
            np.load(truth_path, allow_pickle=False) as truth,
        ):
            expected = np.arange(plan.target_start, plan.target_stop)
            if (
                tuple(data["station_ids"]) != plan.station_ids
                or tuple(data["model_names"]) != MODEL_NAMES
                or not np.array_equal(data["target_indices"], expected)
            ):
                raise DataContractError("Unexpected tensor sample IDs or model order")
            if (
                str(data["plan_sha256"]),
                str(data["run_id"]),
                str(data["scenario_id"]),
                float(data["wait_minutes"]),
            ) != (plan.fingerprint, plan.run_id, plan.scenario_id, plan.wait_minutes):
                raise DataContractError("Mixed tensor plan/run/scenario/wait context")
            indices = truth["target_indices"]
            if (
                indices.ndim != 1
                or len(indices) < len(expected)
                or not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
                or tuple(truth["station_ids"]) != plan.station_ids
            ):
                raise DataContractError("Unexpected truth grid/station IDs")
            start = plan.target_start - int(indices[0])
            if start < 0 or start + len(expected) > len(indices):
                raise DataContractError("Truth does not cover expected targets")
            selected = slice(start, start + len(expected))
            return score_arrays(
                data["predictions"],
                truth["values"][selected],
                truth["valid"][selected],
                plan,
                data["slices"],
                bootstrap=bootstrap,
            )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        if isinstance(exc, DataContractError):
            raise
        raise DataContractError(f"Invalid prediction/truth archive: {exc}") from exc


def select_regime(cases):
    if not cases:
        raise DataContractError("Regime selection requires scored validation cases")
    scores = {}
    for regime in REGIMES:
        grouped = {}
        for case in cases:
            values = [
                case["metrics"]["models"][f"{regime}/{name}"]["mae"]
                for name in ("standard", "quality_aware")
            ]
            if any(value is None or not np.isfinite(value) for value in values):
                raise DataContractError("Regime selection requires finite scored validation MAE")
            value = np.mean(values)
            grouped.setdefault(case["scenario_name"], []).append(float(value))
        scores[regime] = float(np.mean([np.mean(values) for values in grouped.values()]))
    return {
        "selected_regime": "augmented" if scores["augmented"] < scores["clean"] else "clean",
        "scores": scores,
        "rule": "equal scenario mean of seed/wait mean MAE of Standard and full quality; tie clean",
    }
