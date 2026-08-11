"""
Trains and compares Logistic Regression and Random Forest models on the
churn dataset, then saves the best-performing pipeline (preprocessing +
model bundled together) to models/churn_model.joblib.

Run:
    python src/train.py
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
METADATA_PATH = "models/model_metadata.json"

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, pos_label="Yes"), 4),
        "roc_auc": round(roc_auc_score((y_test == "Yes").astype(int), y_proba), 4),
    }


def main():
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"{name}: {metrics}")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    best_metrics = results[best_name]

    print(f"\nBest model: {best_name} -> {best_metrics}")

    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "model_type": best_name,
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": best_metrics,
        "all_candidates": results,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
