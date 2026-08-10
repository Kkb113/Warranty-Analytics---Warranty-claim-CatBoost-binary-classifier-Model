"""Run the predetermined Phase 9 development experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from catboost import CatBoostError

from ..feature_mart.manifest import sha256_file
from .adapters import adapt_matrix, split_development_frame
from .catboost_baseline import (
    effective_parameters,
    fit_classifier,
    predict_probabilities,
    save_model,
)
from .metrics import calculate_metrics, prevalence_probabilities, probability_sha256
from .models import (
    BaselineModelError,
    BaselineSettings,
    DevelopmentTargets,
    ExperimentResult,
    FeatureSetSpec,
)
from .target import KEY, TARGET


def _aligned_target(feature_rows: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    aligned = feature_rows[[KEY]].merge(targets, on=KEY, how="left", validate="one_to_one")
    if aligned[TARGET].isna().any() or len(aligned) != len(feature_rows):
        raise BaselineModelError("Development feature/target alignment failed.")
    return cast(np.ndarray, aligned[TARGET].to_numpy(dtype="int8"))


def run_experiments(
    development: pd.DataFrame,
    targets: DevelopmentTargets,
    feature_sets: dict[str, FeatureSetSpec],
    settings: BaselineSettings,
    model_dir: Path,
) -> list[ExperimentResult]:
    """Fit E1-E4 on TRAIN only and score E0-E4 on VALIDATION only."""

    train_rows, validation_rows = split_development_frame(development)
    y_train = _aligned_target(train_rows, targets.train)
    y_validation = _aligned_target(validation_rows, targets.validation)
    constant = prevalence_probabilities(y_train, len(validation_rows))
    results = [
        ExperimentResult(
            experiment_id="E0",
            model_type="constant_train_prevalence",
            status="SUCCESS",
            feature_set=None,
            metrics=calculate_metrics(y_validation, constant, threshold=settings.fixed_threshold),
            probabilities=pd.Series(constant, index=validation_rows.index),
            validation_probability_sha256=probability_sha256(constant),
        )
    ]
    for experiment_id in ("E1", "E2", "E3", "E4"):
        spec = feature_sets[experiment_id]
        train_matrix = adapt_matrix(train_rows, spec, settings)
        validation_matrix = adapt_matrix(validation_rows, spec, settings)
        try:
            started = time.perf_counter()
            model = fit_classifier(train_matrix, y_train, spec, settings)
            training_seconds = time.perf_counter() - started
            started = time.perf_counter()
            probabilities = predict_probabilities(model, validation_matrix, spec)
            prediction_seconds = time.perf_counter() - started
        except CatBoostError as exc:
            if experiment_id != "E4":
                raise BaselineModelError(
                    f"{experiment_id} CatBoost training failed: {exc}"
                ) from exc
            results.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    model_type="CatBoostClassifier_native_text",
                    status="UNAVAILABLE_WITH_WARNING",
                    feature_set=spec,
                    metrics={},
                    probabilities=None,
                    warning=f"Native text corpus unavailable to CatBoost: {exc}",
                )
            )
            continue
        model_path = model_dir / f"{experiment_id.casefold()}.cbm"
        save_model(model, model_path)
        results.append(
            ExperimentResult(
                experiment_id=experiment_id,
                model_type="CatBoostClassifier_native_text"
                if spec.text_features
                else "CatBoostClassifier",
                status="SUCCESS",
                feature_set=spec,
                metrics=calculate_metrics(
                    y_validation, probabilities, threshold=settings.fixed_threshold
                ),
                probabilities=pd.Series(probabilities, index=validation_rows.index),
                model=model,
                model_file=f"models/{model_path.name}",
                model_sha256=sha256_file(model_path),
                training_seconds=training_seconds,
                prediction_seconds=prediction_seconds,
                effective_parameters=effective_parameters(model),
                validation_probability_sha256=probability_sha256(probabilities),
            )
        )
    return results
