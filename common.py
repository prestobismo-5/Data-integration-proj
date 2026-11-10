"""Shared helpers."""
import pandas as pd

# Shared file locations and settings used by the other scripts.
DB_PATH = "data/health.db"
CHROMA_PATH = "data/chroma"
SEED = 42
# These columns are the values the model uses to make a prediction.
FEATURES = ["age", "sex_m", "bmi", "systolic_bp", "diastolic_bp",
            "glucose_mgdl", "cholesterol_mgdl", "smoker"]


def describe(r):
    """Turn a patient row (dict/Series) into a short text profile for embedding."""
    # Build a readable description from the measurements that are available.
    bits = []
    if pd.notna(r.get("age")):
        bits.append(f"{int(r['age'])}-year-old")
    bits.append({"M": "male", "F": "female"}.get(r.get("sex"), "person"))
    for col, label, fmt in [("bmi", "BMI", ".1f"), ("systolic_bp", "systolic BP", ".0f"),
                            ("diastolic_bp", "diastolic BP", ".0f"),
                            ("glucose_mgdl", "glucose", ".0f"),
                            ("cholesterol_mgdl", "cholesterol", ".0f")]:
        v = r.get(col)
        if v is not None and pd.notna(v):
            bits.append(f"{label} {format(float(v), fmt)}")
    if r.get("smoker") == 1:
        bits.append("smoker")
    return ", ".join(bits)
