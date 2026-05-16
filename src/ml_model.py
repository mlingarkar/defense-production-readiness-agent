from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "program",
    "part_category",
    "production_cell",
    "priority_level",
    "status",
    "material_available",
    "supplier_delay_days",
    "inspection_hold",
    "quality_hold",
    "rework_hours",
    "defect_count",
    "machine_availability_pct",
    "labor_capacity_pct",
    "estimated_cost_impact",
]

CATEGORICAL_FEATURES = [
    "program",
    "part_category",
    "production_cell",
    "priority_level",
    "status",
]

NUMERIC_FEATURES = [
    "material_available",
    "supplier_delay_days",
    "inspection_hold",
    "quality_hold",
    "rework_hours",
    "defect_count",
    "machine_availability_pct",
    "labor_capacity_pct",
    "estimated_cost_impact",
]


def create_late_delivery_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target for late delivery.

    A work order is labeled as late if the forecast completion date is after
    the planned completion date.
    """
    modeled_df = df.copy()

    modeled_df["planned_completion_date"] = pd.to_datetime(
        modeled_df["planned_completion_date"]
    )
    modeled_df["forecast_completion_date"] = pd.to_datetime(
        modeled_df["forecast_completion_date"]
    )

    modeled_df["late_delivery_flag"] = (
        modeled_df["forecast_completion_date"]
        > modeled_df["planned_completion_date"]
    ).astype(int)

    return modeled_df


def train_late_delivery_model(
    df: pd.DataFrame,
    model_output_path: str = "outputs/models/late_delivery_model.joblib",
) -> tuple[pd.DataFrame, dict]:
    """
    Train a machine learning model to predict late delivery risk.

    The model uses synthetic production readiness data and is intended for
    portfolio demonstration purposes.
    """
    modeled_df = create_late_delivery_target(df)

    X = modeled_df[FEATURE_COLUMNS]
    y = modeled_df["late_delivery_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_split=8,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    test_predictions = pipeline.predict(X_test)
    all_probabilities = pipeline.predict_proba(X)[:, 1]

    modeled_df["late_delivery_probability"] = all_probabilities
    modeled_df["late_delivery_prediction"] = (
        modeled_df["late_delivery_probability"] >= 0.50
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "late_delivery_rate": y.mean(),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            test_predictions,
            output_dict=True,
        ),
    }

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return modeled_df, metrics


def write_model_report(
    metrics: dict,
    output_path: str = "outputs/reports/late_delivery_model_report.md",
) -> None:
    """
    Write model performance metrics to a markdown report.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = metrics["confusion_matrix"]

    content = f"""# Late Delivery Prediction Model Report

## Model Objective

This model predicts whether a synthetic defense production work order is likely to finish later than its planned completion date.

## Model Type

Random Forest Classifier with preprocessing for categorical and numeric production features.

## Training Summary

- Training rows: {metrics["training_rows"]}
- Testing rows: {metrics["testing_rows"]}
- Baseline late-delivery rate: {metrics["late_delivery_rate"]:.2%}
- Accuracy: {metrics["accuracy"]:.2%}

## Confusion Matrix

|  | Predicted On Time | Predicted Late |
|---|---:|---:|
| Actual On Time | {matrix[0][0]} | {matrix[0][1]} |
| Actual Late | {matrix[1][0]} | {matrix[1][1]} |

## Model Notes

The model is trained entirely on synthetic data and is intended for portfolio demonstration. It is not designed for real production, quality, engineering, safety, or defense decision-making.
"""

    report_path.write_text(content, encoding="utf-8")