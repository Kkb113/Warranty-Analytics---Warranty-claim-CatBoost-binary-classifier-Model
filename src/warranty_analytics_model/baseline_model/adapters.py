"""Deterministic Phase 9 joins and CatBoost model-adapter mappings."""

from __future__ import annotations

import pandas as pd

from .models import BaselineModelError, BaselineSettings, FeatureSetSpec, Phase9Inputs

KEY = "warranty_claim_key"


def build_development_feature_frame(
    inputs: Phase9Inputs,
    feature_sets: dict[str, FeatureSetSpec],
) -> pd.DataFrame:
    """Join Phase 7/8 by key one-to-one and return TRAIN/VALIDATION rows only."""

    maximum = feature_sets["E4"].feature_names
    phase7_names = set(inputs.phase7_lineage)
    structured_columns = [KEY] + [name for name in maximum if name in phase7_names]
    text_columns = [KEY] + [name for name in maximum if name not in phase7_names]
    structured = inputs.structured_features[structured_columns]
    text = inputs.text_features[text_columns]
    if structured[KEY].duplicated().any() or text[KEY].duplicated().any():
        raise BaselineModelError("Phase 7 or Phase 8 feature keys are duplicated.")
    combined = structured.merge(text, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (combined["_merge"] != "both").any():
        raise BaselineModelError("Phase 7 and Phase 8 memberships differ.")
    combined = combined.drop(columns="_merge")
    controls = inputs.assignments[[KEY, "split"]]
    joined = controls.merge(combined, on=KEY, how="left", validate="one_to_one", indicator=True)
    if (joined["_merge"] != "both").any() or len(joined) != len(controls):
        raise BaselineModelError("Feature membership differs from Phase 6 assignments.")
    joined = joined.drop(columns="_merge")
    development = joined.loc[joined["split"].isin(["TRAIN", "VALIDATION"])].copy()
    if (development["split"] == "TEST").any():
        raise BaselineModelError("TEST features entered a development matrix.")
    return development.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def adapt_matrix(
    frame: pd.DataFrame,
    feature_set: FeatureSetSpec,
    settings: BaselineSettings,
) -> pd.DataFrame:
    """Apply fixed model-only representations in deterministic feature order."""

    missing = sorted(set(feature_set.feature_names) - set(frame.columns))
    if missing:
        raise BaselineModelError("Model matrix is missing features: " + ", ".join(missing))
    matrix = frame.loc[:, list(feature_set.feature_names)].copy()
    for name in feature_set.numeric_features:
        matrix[name] = pd.to_numeric(matrix[name], errors="coerce")
    for name in feature_set.boolean_features:
        values = pd.to_numeric(matrix[name], errors="coerce")
        invalid = values.notna() & ~values.isin([0, 1])
        if invalid.any():
            raise BaselineModelError(f"Boolean feature contains a non-binary value: {name}")
        matrix[name] = values.astype("float64")
    for name in feature_set.categorical_features:
        matrix[name] = (
            matrix[name].astype("string").fillna(settings.categorical_missing_value).astype(str)
        )
    for name in feature_set.text_features:
        matrix[name] = matrix[name].astype("string").fillna(settings.text_missing_value).astype(str)
    if tuple(matrix.columns) != feature_set.feature_names:
        raise BaselineModelError("Phase 9 feature order changed during adaptation.")
    return matrix


def split_development_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"] == "TRAIN"].copy()
    validation = frame.loc[frame["split"] == "VALIDATION"].copy()
    if train.empty or validation.empty:
        raise BaselineModelError("TRAIN and VALIDATION feature matrices must be nonempty.")
    return train, validation
