"""Fixed CatBoost construction, fitting, persistence, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .models import BaselineSettings, FeatureSetSpec


def build_pool(
    matrix: pd.DataFrame,
    feature_set: FeatureSetSpec,
    target: np.ndarray | None = None,
) -> Pool:
    """Build a Pool with explicit names so type routing is independently auditable."""

    return Pool(
        data=matrix,
        label=target,
        feature_names=list(feature_set.feature_names),
        cat_features=list(feature_set.categorical_features),
        text_features=list(feature_set.text_features),
    )


def fit_classifier(
    train_matrix: pd.DataFrame,
    train_target: np.ndarray,
    feature_set: FeatureSetSpec,
    settings: BaselineSettings,
) -> CatBoostClassifier:
    parameters = dict(settings.catboost_parameters)
    if feature_set.text_features:
        parameters["text_processing"] = settings.text_processing
    model = CatBoostClassifier(**parameters)
    model.fit(build_pool(train_matrix, feature_set, train_target))
    return model


def predict_probabilities(
    model: CatBoostClassifier,
    matrix: pd.DataFrame,
    feature_set: FeatureSetSpec,
) -> np.ndarray:
    return np.asarray(model.predict_proba(build_pool(matrix, feature_set))[:, 1], dtype="float64")


def save_model(model: CatBoostClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path), format="cbm")


def load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path), format="cbm")
    return model


def effective_parameters(model: CatBoostClassifier) -> dict[str, Any]:
    return {str(key): value for key, value in model.get_all_params().items()}
