"""Train the risk model and save its evaluation metrics.

Run from the project folder: python src/train_model.py
"""
import json
import sqlite3

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from common import DB_PATH, FEATURES, SEED


def load_patient_data():
    """Read labeled patient records from the SQLite database."""
    with sqlite3.connect(DB_PATH) as con:
        data = pd.read_sql(
            "SELECT * FROM patient_record WHERE risk_label IS NOT NULL", con
        )

    # The model expects sex as a number: 1 for male and 0 for other values.
    data["sex_m"] = (data["sex"] == "M").astype(float)
    return data


def calculate_group_metrics(test_data):
    """Measure model results for groups with enough examples of both labels."""
    group_metrics = []
    nhanes_data = test_data[test_data["source_dataset"] == "NHANES 2017-18"]
    columns = ["source_dataset", "sex", "age_band", "race_ethnicity"]

    for column in columns:
        # Race categories are only available for the NHANES records.
        if column == "race_ethnicity":
            data_to_group = nhanes_data
        else:
            data_to_group = test_data

        for group_name, group_data in data_to_group.groupby(column):
            has_both_labels = group_data["risk_label"].nunique() == 2
            # Skip small groups and groups with only positive or only negative labels.
            if len(group_data) < 30 or not has_both_labels:
                continue

            group_metrics.append({
                "attribute": column,
                "group": str(group_name),
                "n": len(group_data),
                "auc": round(
                    roc_auc_score(group_data["risk_label"], group_data["p"]), 3
                ),
                "actual_rate": round(group_data["risk_label"].mean(), 3),
                "mean_predicted": round(group_data["p"].mean(), 3),
            })

    return group_metrics


def main():
    data = load_patient_data()

    # Keep one quarter of the records aside to check the model on data
    # it did not train on.
    train_data, test_data = train_test_split(
        data,
        test_size=0.25,
        stratify=data["risk_label"],
        random_state=SEED,
    )

    # Fit the model using only the training records.
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=SEED,
    )
    model.fit(train_data[FEATURES], train_data["risk_label"])

    # Estimate the chance of a positive label for each held-out record.
    test_data = test_data.copy()
    test_data["p"] = model.predict_proba(test_data[FEATURES])[:, 1]

    metrics = {
        "overall": {
            "n": len(test_data),
            "auc": roc_auc_score(test_data["risk_label"], test_data["p"]),
            "brier": brier_score_loss(test_data["risk_label"], test_data["p"]),
        },
        "groups": calculate_group_metrics(test_data),
    }

    # Save the trained model for the app and its evaluation results for review.
    joblib.dump({"model": model, "features": FEATURES}, "data/model.joblib")
    with open("data/metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
