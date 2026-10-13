
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in diabetes.csv: {missing}"
        )

    df = df[required].copy()
    df = df.drop_duplicates()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # In this dataset, zero is commonly treated as missing
    # for these measurements. Verify against your data dictionary.
    for col in [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]:
        df.loc[df[col] == 0, col] = float("nan")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need enough examples of both target classes.")

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                FEATURES,
            )
        ]
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("Diabetes evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "feature_importance": dict(zip(
            FEATURES,
            model.named_steps["classifier"].feature_importances_,
        )),
    }

    joblib.dump(bundle, MODEL_DIR / "diabetes_model.pkl")
    print("Saved models/diabetes_model.pkl")


if __name__ == "__main__":
    main()
