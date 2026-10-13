
from pathlib import Path

import joblib
import pandas as pd

from flask_smorest import Blueprint
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "Age", "RestingBP", "Cholesterol",
    "FastingBS", "MaxHR", "Oldpeak",
]

CATEGORICAL_FEATURES = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "HeartDisease"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in heart.csv: {missing}"
        )

    df = df[required].copy().drop_duplicates()

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need enough examples of both target classes.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

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

    print("Heart evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    transformed_names = (
        model.named_steps["preprocess"].get_feature_names_out()
    )
    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "feature_importance": dict(zip(
            transformed_names.tolist(),
            importances.tolist(),
        )),
    }

    joblib.dump(bundle, MODEL_DIR / "heart_model.pkl")
    print("Saved models/heart_model.pkl")


if __name__ == "__main__":
    main()
