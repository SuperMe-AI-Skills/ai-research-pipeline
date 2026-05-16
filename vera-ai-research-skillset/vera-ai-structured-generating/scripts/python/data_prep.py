# -*- coding: utf-8 -*-
"""data_prep.py

Data preparation utilities for structured/tabular data analysis:
- Missing value imputation
- Categorical encoding (one-hot / label)
- Feature scaling (standard / minmax / robust)
- Train / validation / test splitting
- Task type detection
- Full feature preparation pipeline
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)


# -------------------------------------------------------------------
# 1. Missing value imputation
# -------------------------------------------------------------------

def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Fill NaN values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (modified in-place on a copy).
    numeric_strategy : str
        Strategy for numeric columns: 'median', 'mean', or 'zero'.
    categorical_strategy : str
        Strategy for categorical columns: 'mode' or 'missing'.

    Returns
    -------
    df_imputed : pd.DataFrame
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric imputation
    for col in numeric_cols:
        if df[col].isna().any():
            if numeric_strategy == "median":
                fill_val = df[col].median()
            elif numeric_strategy == "mean":
                fill_val = df[col].mean()
            elif numeric_strategy == "zero":
                fill_val = 0.0
            else:
                raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}")
            df[col] = df[col].fillna(fill_val)

    # Categorical imputation
    for col in categorical_cols:
        if df[col].isna().any():
            if categorical_strategy == "mode":
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
            elif categorical_strategy == "missing":
                fill_val = "MISSING"
            else:
                raise ValueError(f"Unknown categorical_strategy: {categorical_strategy}")
            df[col] = df[col].fillna(fill_val)

    return df


# -------------------------------------------------------------------
# 2. Categorical encoding
# -------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    method: str = "onehot",
    max_categories: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical columns using one-hot or label encoding.

    - One-hot for columns with <= max_categories unique values.
    - Label encoding for columns with > max_categories unique values
      (or when method='label').

    Parameters
    ----------
    df : pd.DataFrame
    categorical_cols : list of column names to encode
    method : 'onehot' (auto-select based on cardinality) or 'label'
    max_categories : threshold for one-hot vs label encoding

    Returns
    -------
    df_encoded : pd.DataFrame with encoded columns
    encoder_dict : dict mapping column name -> fitted encoder
    """
    df = df.copy()
    encoder_dict: Dict[str, Any] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()

        if method == "label" or n_unique > max_categories:
            # Label encoding for high-cardinality
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoder_dict[col] = {"type": "label", "encoder": le}
        else:
            # One-hot encoding for low-cardinality
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df[[col]])
            feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)
            encoder_dict[col] = {"type": "onehot", "encoder": ohe}

    return df, encoder_dict


# -------------------------------------------------------------------
# 3. Feature scaling
# -------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Fit scaler on train, transform all splits.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    method : 'standard', 'minmax', or 'robust'

    Returns
    -------
    X_train_s, X_val_s, X_test_s : scaled arrays
    scaler : fitted scaler object
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s, scaler


# -------------------------------------------------------------------
# 4. Train / validation / test split
# -------------------------------------------------------------------

def split_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.20,
    val_size_of_train: float = 0.25,
    random_state: int = 2025,
    stratify: bool = True,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train, validation, and test sets.

    - Stratified split for classification (when stratify=True).
    - Group-aware split when group_col is provided.
    - Random split for regression.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : name of the target column
    test_size : fraction for test set (from total)
    val_size_of_train : fraction for validation (from remaining after test)
    random_state : RNG seed
    stratify : whether to use stratified splitting
    group_col : optional column for group-aware splitting

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrames
    """
    task = detect_task_type(df[target_col])

    if group_col is not None:
        # Group-aware split
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_val_idx, test_idx = next(
            gss_test.split(df, groups=df[group_col])
        )
        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_of_train, random_state=random_state
        )
        train_idx, val_idx = next(
            gss_val.split(df_train_val, groups=df_train_val[group_col])
        )
        df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    else:
        # Standard split
        stratify_col = df[target_col] if (stratify and task == "classification") else None

        df_train_val, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )

        stratify_tv = (
            df_train_val[target_col]
            if (stratify and task == "classification")
            else None
        )

        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size_of_train,
            random_state=random_state,
            stratify=stratify_tv,
        )

        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

    print(f"[Split] Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return df_train, df_val, df_test


# -------------------------------------------------------------------
# 5. Task type detection
# -------------------------------------------------------------------

def detect_task_type(y: Union[pd.Series, np.ndarray]) -> str:
    """
    Detect whether a target variable represents classification or regression.

    Uses multiple signals:
    - dtype: object/category/bool → classification
    - float dtype with non-integer values → regression
    - integer-valued: ≤10 unique → classification, >20 → regression,
      10-20 → classification only if values look like codes (small range)
    """
    if isinstance(y, pd.Series):
        if y.dtype == "object" or y.dtype.name == "category" or y.dtype == bool:
            return "classification"
        y_arr = y.dropna().values
    else:
        y_arr = y[~np.isnan(y)] if np.issubdtype(np.asarray(y).dtype, np.floating) else np.asarray(y)

    y_arr = np.asarray(y_arr)
    n_unique = len(np.unique(y_arr))

    # Clearly categorical
    if n_unique <= 10:
        return "classification"

    # Clearly continuous
    if n_unique > 20:
        return "regression"

    # Ambiguous zone (10-20 unique): check if values are float with fractional parts
    if np.issubdtype(y_arr.dtype, np.floating):
        has_fractions = not np.allclose(y_arr, np.round(y_arr))
        if has_fractions:
            return "regression"

    # Integer-valued with 10-20 unique: likely classification if range is small
    val_range = float(np.max(y_arr) - np.min(y_arr))
    if val_range <= 20:
        return "classification"

    return "regression"


# -------------------------------------------------------------------
# 6. Feature preparation pipeline (fit/transform pattern)
# -------------------------------------------------------------------

def _drop_and_separate(
    df: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Internal helper: drop target + IDs, detect categoricals."""
    df = df.copy()
    y = df[target_col].values
    df = df.drop(columns=[target_col])

    if id_cols is not None:
        df = df.drop(columns=[c for c in id_cols if c in df.columns])

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    return df, y, categorical_cols


def fit_prepare_features(
    df_train: pd.DataFrame,
    target_col: str,
    id_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    encoding_method: str = "onehot",
    max_categories: int = 10,
    skip_encoding_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Fit imputation and encoding on TRAINING data only, then transform it.

    This avoids holdout leakage — val/test must use transform_features()
    with the encoders returned here.

    Parameters
    ----------
    df_train : pd.DataFrame — training split only
    target_col : name of the target column
    id_cols : columns to drop (IDs, timestamps, etc.)
    categorical_cols : explicit list of categorical columns (auto-detected if None)
    numeric_strategy : 'median', 'mean', or 'zero'
    categorical_strategy : 'mode' or 'missing'
    encoding_method : 'onehot' or 'label'
    max_categories : cardinality threshold for one-hot vs label encoding
    skip_encoding_cols : categorical columns to leave as-is (e.g., for CatBoost)

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    feature_names : list of feature column names
    prep_state : dict containing fitted encoders and imputation stats
    """
    df, y, categorical_cols = _drop_and_separate(
        df_train, target_col, id_cols, categorical_cols
    )

    # Compute imputation values from training data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    impute_values: Dict[str, Any] = {}
    for col in numeric_cols:
        if df[col].isna().any():
            if numeric_strategy == "median":
                impute_values[col] = df[col].median()
            elif numeric_strategy == "mean":
                impute_values[col] = df[col].mean()
            elif numeric_strategy == "zero":
                impute_values[col] = 0.0
            else:
                raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}")
            df[col] = df[col].fillna(impute_values[col])

    cat_cols_in_data = [c for c in df.select_dtypes(include=["object", "category"]).columns
                        if c in categorical_cols]
    cat_impute_values: Dict[str, Any] = {}
    for col in cat_cols_in_data:
        if df[col].isna().any():
            if categorical_strategy == "mode":
                mode_val = df[col].mode()
                cat_impute_values[col] = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
            elif categorical_strategy == "missing":
                cat_impute_values[col] = "MISSING"
            else:
                raise ValueError(f"Unknown categorical_strategy: {categorical_strategy}")
            df[col] = df[col].fillna(cat_impute_values[col])

    # Encode categoricals (skip columns reserved for native handling, e.g. CatBoost)
    if skip_encoding_cols is None:
        skip_encoding_cols = []
    cols_to_encode = [c for c in categorical_cols if c in df.columns and c not in skip_encoding_cols]

    encoders: Dict[str, Any] = {}
    if cols_to_encode:
        df, encoders = encode_categoricals(df, cols_to_encode, method=encoding_method,
                                           max_categories=max_categories)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)

    prep_state = {
        "impute_values_numeric": impute_values,
        "impute_values_categorical": cat_impute_values,
        "encoders": encoders,
        "feature_names": feature_names,
        "id_cols": id_cols,
        "target_col": target_col,
        "categorical_cols": categorical_cols,
        "skip_encoding_cols": skip_encoding_cols,
        "encoding_method": encoding_method,
        "max_categories": max_categories,
    }

    print(f"[FitPrepFeatures] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[FitPrepFeatures] Features: {len(feature_names)}")

    return X, y, feature_names, prep_state


def transform_features(
    df: pd.DataFrame,
    prep_state: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform validation or test data using encoders fitted on training data.

    Parameters
    ----------
    df : pd.DataFrame — val or test split
    prep_state : dict returned by fit_prepare_features()

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    target_col = prep_state["target_col"]
    id_cols = prep_state["id_cols"]
    categorical_cols = prep_state["categorical_cols"]
    skip_encoding_cols = prep_state["skip_encoding_cols"]

    df, y, _ = _drop_and_separate(df, target_col, id_cols, categorical_cols)

    # Apply imputation using training statistics
    for col, val in prep_state["impute_values_numeric"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in prep_state["impute_values_categorical"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Apply encoding using fitted encoders
    encoders = prep_state["encoders"]
    for col, enc_info in encoders.items():
        if col not in df.columns:
            continue
        if enc_info["type"] == "label":
            le = enc_info["encoder"]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
        elif enc_info["type"] == "onehot":
            ohe = enc_info["encoder"]
            encoded = ohe.transform(df[[col]])
            feature_names_ohe = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names_ohe, index=df.index)
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)

    # Ensure columns match training feature order
    expected_features = prep_state["feature_names"]
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[expected_features]

    X = df.values.astype(np.float32)
    return X, y


def get_catboost_cat_indices(
    feature_names: List[str],
    categorical_cols: List[str],
) -> List[int]:
    """
    Return column indices of categorical features for CatBoost's cat_features parameter.

    Use this after fit_prepare_features(skip_encoding_cols=categorical_cols)
    to get the indices that CatBoost should treat as native categoricals.

    Parameters
    ----------
    feature_names : list of str — from fit_prepare_features()
    categorical_cols : list of str — columns that were NOT encoded

    Returns
    -------
    list of int — positional indices into the feature matrix
    """
    return [i for i, name in enumerate(feature_names) if name in categorical_cols]


