import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline import NON_FEATURE_COLUMNS, STAGE_LABELS

logger = logging.getLogger(__name__)


def load_feature_matrix(path: Path) -> pd.DataFrame:
    path = Path(path)
    try:
        matrix = pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature matrix not found: {path}. Run the extraction stage first.") from None
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(f"Could not read feature matrix {path}: {exc}") from exc

    missing = {"timestamp", "stage"} - set(matrix.columns)
    if missing:
        raise ValueError(f"Feature matrix is missing required columns: {sorted(missing)}")
    return matrix.sort_values("timestamp").reset_index(drop=True)


def chronological_split(
    matrix: pd.DataFrame,
    train_fraction: float = 0.70,
    gap: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split applied inside each health stage.

    A plain "first 70% / last 30%" cut is wrong here: stages are contiguous in
    time, so the tail of the run would be almost pure Severe_Fault and the
    model would never see that class in training. Cutting each stage at its
    own 70% mark keeps every class represented while still guaranteeing that
    no test snapshot precedes a training snapshot of the same stage.

    Consecutive snapshots are ten minutes apart and strongly autocorrelated,
    so `gap` snapshots after each cut are dropped; otherwise the first test
    rows are near-duplicates of the last training rows and the score is
    inflated.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    for stage in STAGE_LABELS:
        block = matrix[matrix["stage"] == stage].sort_values("timestamp")
        if block.empty:
            logger.warning("Stage %s has no samples; skipping", stage)
            continue
        cut = int(len(block) * train_fraction)
        train_parts.append(block.iloc[:cut])
        test_parts.append(block.iloc[cut + gap:])

    if not train_parts:
        raise ValueError("No labelled samples available for training")

    train, test = pd.concat(train_parts), pd.concat(test_parts)
    if train["stage"].nunique() < 2:
        raise ValueError("Training split contains fewer than two classes")
    if test.empty:
        raise ValueError("Test split is empty; reduce `gap` or train_fraction")
    return train, test


def train_and_evaluate(
    matrix: pd.DataFrame,
    model_path: Path,
    report_path: Path,
    train_fraction: float = 0.70,
    gap: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    # run_progress is the label in disguise (stage is a threshold on it), so
    # it must never reach the model.
    feature_columns = [c for c in matrix.columns if c not in NON_FEATURE_COLUMNS]
    train, test = chronological_split(matrix, train_fraction, gap)
    logger.info("Train rows: %d | Test rows: %d | Features: %d", len(train), len(test), len(feature_columns))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("forest", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    model.fit(train[feature_columns], train["stage"])

    predictions = model.predict(test[feature_columns])
    labels = [s for s in STAGE_LABELS if s in set(train["stage"]) | set(test["stage"])]

    cm = pd.DataFrame(
        confusion_matrix(test["stage"], predictions, labels=labels),
        index=[f"true_{s}" for s in labels],
        columns=[f"pred_{s}" for s in labels],
    )
    report = classification_report(test["stage"], predictions, labels=labels, zero_division=0)
    macro_f1 = f1_score(test["stage"], predictions, labels=labels, average="macro", zero_division=0)

    importances = pd.Series(
        model.named_steps["forest"].feature_importances_, index=feature_columns
    ).sort_values(ascending=False)

    summary = "\n".join([
        "Confusion matrix (rows = actual, columns = predicted)",
        cm.to_string(),
        "",
        "Classification report",
        report,
        f"Macro F1: {macro_f1:.4f}",
        "",
        "Top 10 features by importance",
        importances.head(10).round(4).to_string(),
    ])
    print(summary)

    try:
        model_path, report_path = Path(model_path), Path(report_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": feature_columns}, model_path)
        report_path.write_text(summary + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not persist training artefacts: {exc}") from exc

    return {"macro_f1": float(macro_f1), "confusion_matrix": cm, "feature_importances": importances}