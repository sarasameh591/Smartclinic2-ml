
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import Literal
from flask_smorest import Api, Blueprint
from marshmallow import Schema, fields

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

diabetes_bundle = joblib.load(MODEL_DIR / "diabetes_model.pkl")
heart_bundle = joblib.load(MODEL_DIR / "heart_model.pkl")

diabetes_model = diabetes_bundle["model"]
heart_model = heart_bundle["model"]

diabetes_features = diabetes_bundle["features"]
heart_features = heart_bundle["features"]

app = Flask(__name__)
app.config["API_TITLE"] = "Smart Clinic ML API"
app.config["API_VERSION"] = "1.0.0"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/docs"
app.config["OPENAPI_SWAGGER_UI_URL"] = (
    "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
)
api = Api(app)

diabetes_bp = Blueprint(
    "diabetes", "diabetes",
    url_prefix="/predict/diabetes",
    description="Diabetes risk prediction",
)
heart_bp = Blueprint(
    "heart", "heart",
    url_prefix="/predict/heart",
    description="Heart disease risk prediction",
)


# Diabetes: 8 features
class DiabetesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: int = Field(ge=0, le=30)
    Glucose: float = Field(gt=0, le=500)
    BloodPressure: float = Field(gt=0, le=300)
    SkinThickness: float = Field(ge=0, le=150)
    Insulin: float = Field(ge=0, le=3000)
    BMI: float = Field(gt=0, le=100)
    DiabetesPedigreeFunction: float = Field(ge=0, le=10)
    Age: int = Field(ge=1, le=120)


# Heart: 11 features
class HeartInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(ge=1, le=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float = Field(ge=0, le=300)
    Cholesterol: float = Field(ge=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(ge=0, le=300)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-10, le=20)
    ST_Slope: Literal["Up", "Flat", "Down"]


def risk_label(score):
    if score > 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def predict(bundle, features, payload):
    model = bundle["model"]
    row = pd.DataFrame([payload], columns=features)

    # Probability of class 1, expressed as a percentage.
    class_one_index = list(model.classes_).index(1)
    score = float(model.predict_proba(row)[0][class_one_index] * 100)

    importance = bundle.get("feature_importance", {})
    top_factors = sorted(
        importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return {
        "risk_score": round(score, 2),
        "risk_level": risk_label(score),
        "predicted_class": int(model.predict(row)[0]),
        "top_model_features": [
            {"feature": name, "importance": round(float(value), 4)}
            for name, value in top_factors
        ],
        "disclaimer": (
            "For educational decision support only; "
            "not a medical diagnosis."
        ),
    }


@diabetes_bp.route("/", methods=["POST"])
@diabetes_bp.arguments(Schema.from_dict({
    "Pregnancies": fields.Integer(required=True),
    "Glucose": fields.Float(required=True),
    "BloodPressure": fields.Float(required=True),
    "SkinThickness": fields.Float(required=True),
    "Insulin": fields.Float(required=True),
    "BMI": fields.Float(required=True),
    "DiabetesPedigreeFunction": fields.Float(required=True),
    "Age": fields.Integer(required=True),
}))
def diabetes_predict(raw_data):
    try:
        validated = DiabetesInput.model_validate(raw_data)
    except ValidationError as exc:
        return jsonify({"error": "Invalid diabetes input",
                        "details": exc.errors()}), 400

    return jsonify(predict(
        diabetes_bundle,
        diabetes_features,
        validated.model_dump(),
    ))


@heart_bp.route("/", methods=["POST"])
@heart_bp.arguments(Schema.from_dict({
    "Age": fields.Integer(required=True),
    "Sex": fields.String(required=True),
    "ChestPainType": fields.String(required=True),
    "RestingBP": fields.Float(required=True),
    "Cholesterol": fields.Float(required=True),
    "FastingBS": fields.Integer(required=True),
    "RestingECG": fields.String(required=True),
    "MaxHR": fields.Float(required=True),
    "ExerciseAngina": fields.String(required=True),
    "Oldpeak": fields.Float(required=True),
    "ST_Slope": fields.String(required=True),
}))
def heart_predict(raw_data):
    try:
        validated = HeartInput.model_validate(raw_data)
    except ValidationError as exc:
        return jsonify({"error": "Invalid heart input",
                        "details": exc.errors()}), 400

    return jsonify(predict(
        heart_bundle,
        heart_features,
        validated.model_dump(),
    ))


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "models": ["diabetes", "heart"],
    })


api.register_blueprint(diabetes_bp)
api.register_blueprint(heart_bp)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)