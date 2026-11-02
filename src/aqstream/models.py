"""One estimator family, identical hyperparameters, separate feature blocks."""

from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from aqstream.data import DataContractError


class Baselines:
    def __init__(self, *, seed: int, model_params: dict):
        params = dict(
            model_params,
            early_stopping=False,
            validation_fraction=None,
            categorical_features=None,
            random_state=seed,
        )
        self.standard = HistGradientBoostingRegressor(**params)
        self.quality_aware = HistGradientBoostingRegressor(**params)
        self.is_fitted = False
        self.fit_seconds: dict[str, float] = {}

    def fit(self, common: np.ndarray, quality: np.ndarray, labels: np.ndarray) -> None:
        if not len(labels) or not np.isfinite(labels).all():
            raise DataContractError("Model fit requires source-valid training labels")
        for name, features in (
            ("standard", common),
            ("quality_aware", np.column_stack((common, quality))),
        ):
            started = perf_counter()
            getattr(self, name).fit(features, labels)
            self.fit_seconds[name] = perf_counter() - started
        self.is_fitted = True

    def predict(
        self,
        common: np.ndarray,
        quality: np.ndarray,
        persistence: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        # No labels, source truth or future metadata is accepted by this API.
        if not self.is_fitted:
            raise DataContractError("Models must be fitted before predict")
        started = perf_counter()
        predictions = {"persistence": persistence.copy()}
        timings = {"persistence": perf_counter() - started}
        for name, features in (
            ("standard", common),
            ("quality_aware", np.column_stack((common, quality))),
        ):
            started = perf_counter()
            predictions[name] = getattr(self, name).predict(features)
            timings[name] = perf_counter() - started
        if not all(np.isfinite(values).all() for values in predictions.values()):
            raise DataContractError("Models produced nonfinite predictions")
        return predictions, timings
