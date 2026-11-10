"""Join the data sources with PySpark and save the results in SQLite.

Run from the project folder: python src/integrate_spark.py
PySpark requires Java.
"""
import glob
import json
import sqlite3

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from common import DB_PATH

RAW = "data/raw"
TRANSFORM_VERSION = "v1"
# All sources are converted to these shared column types before they are joined.
TYPES = {
    "age": "double",
    "sex": "string",
    "race_ethnicity": "string",
    "bmi": "double",
    "systolic_bp": "double",
    "diastolic_bp": "double",
    "glucose_mgdl": "double",
    "cholesterol_mgdl": "double",
    "smoker": "int",
    "risk_label": "int",
}
COLS = ["source_dataset"] + list(TYPES) + ["label_definition"]


def read_csv(spark, path):
    """Read a CSV file into a Spark DataFrame."""
    return spark.read.csv(path, header=True, inferSchema=True)


def conform_to_common_schema(data, source, label_definition):
    """Give each source the same columns and data types."""
    for column, data_type in TYPES.items():
        if column not in data.columns:
            data = data.withColumn(column, F.lit(None).cast(data_type))
        else:
            data = data.withColumn(column, F.col(column).cast(data_type))

    data = data.withColumn("source_dataset", F.lit(source))
    data = data.withColumn("label_definition", F.lit(label_definition))
    # Keep only the shared columns, in the same order for every source.
    return data.select(COLS)


def load_nhanes(spark):
    """Join NHANES files by participant ID and make a risk label."""
    folder = f"{RAW}/nhanes"
    demo = read_csv(spark, f"{folder}/DEMO_J.csv").select(
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"
    )

    # Add measurements and survey answers from the other NHANES files.
    tables = [
        ("BMX_J", ["SEQN", "BMXBMI"]),
        ("BPX_J", ["SEQN", "BPXSY1", "BPXDI1"]),
        ("GLU_J", ["SEQN", "LBXGLU"]),
        ("TCHOL_J", ["SEQN", "LBXTC"]),
        ("SMQ_J", ["SEQN", "SMQ020"]),
        ("DIQ_J", ["SEQN", "DIQ010"]),
        ("BPQ_J", ["SEQN", "BPQ020"]),
    ]
    for filename, columns in tables:
        table = read_csv(spark, f"{folder}/{filename}.csv").select(columns)
        # A left join keeps participants even if they have no measurement in this file.
        demo = demo.join(table, "SEQN", "left")

    adults = demo.filter(F.col("RIDAGEYR") >= 18)
    adults = adults.withColumn("age", F.col("RIDAGEYR"))
    adults = adults.withColumn(
        "sex",
        F.when(F.col("RIAGENDR") == 1, "M").when(F.col("RIAGENDR") == 2, "F"),
    )
    adults = adults.withColumn("race_ethnicity", F.col("RIDRETH3").cast("string"))
    adults = adults.withColumn("bmi", F.col("BMXBMI"))
    adults = adults.withColumn("systolic_bp", F.col("BPXSY1"))
    adults = adults.withColumn("diastolic_bp", F.col("BPXDI1"))
    adults = adults.withColumn("glucose_mgdl", F.col("LBXGLU"))
    adults = adults.withColumn("cholesterol_mgdl", F.col("LBXTC"))
    adults = adults.withColumn(
        "smoker",
        F.when(F.col("SMQ020") == 1, 1).when(F.col("SMQ020") == 2, 0),
    )

    # A positive label means the participant reported either condition.
    adults = adults.withColumn(
        "risk_label",
        F.when((F.col("DIQ010") == 1) | (F.col("BPQ020") == 1), 1).when(
            F.col("DIQ010").isin(2, 3) & (F.col("BPQ020") == 2), 0
        ),
    )
    # Unknown answers do not give enough information to create a label.
    adults = adults.filter(F.col("risk_label").isNotNull())

    return conform_to_common_schema(
        adults,
        "NHANES 2017-18",
        "told by doctor: diabetes OR hypertension (self-reported)",
    )


def load_heart_data(spark):
    """Prepare the UCI Heart Disease data."""
    heart = read_csv(spark, f"{RAW}/uci_heart.csv")
    heart = heart.withColumn(
        "sex", F.when(F.col("sex") == 1, "M").otherwise("F")
    )
    heart = heart.withColumn("systolic_bp", F.col("trestbps"))
    heart = heart.withColumn("cholesterol_mgdl", F.col("chol"))
    heart = heart.withColumn(
        "risk_label", F.when(F.col("num") > 0, 1).otherwise(0)
    )

    return conform_to_common_schema(
        heart, "UCI Heart Disease", "angiographic heart disease (num > 0)"
    )


def load_pima_data(spark):
    """Prepare the Pima data, where zero means a missing measurement."""
    pima = read_csv(spark, f"{RAW}/pima_diabetes.csv")

    for column in ["glucose", "blood_pressure", "bmi"]:
        pima = pima.withColumn(
            column,
            F.when(F.col(column) == 0, None).otherwise(F.col(column)),
        )

    pima = pima.withColumn("sex", F.lit("F"))
    pima = pima.withColumn("glucose_mgdl", F.col("glucose"))
    pima = pima.withColumn("diastolic_bp", F.col("blood_pressure"))
    pima = pima.withColumn("risk_label", F.col("outcome"))

    return conform_to_common_schema(
        pima, "Pima Diabetes", "diabetes onset within 5 years (outcome = 1)"
    )


def combine_patient_data(sources):
    """Stack the standardized sources and add an ID and age group."""
    # unionByName stacks rows by column name instead of column position.
    patients = sources[0]
    for source in sources[1:]:
        patients = patients.unionByName(source)

    patients = patients.withColumn("record_id", F.monotonically_increasing_id())
    patients = patients.withColumn(
        "age_band",
        F.when(F.col("age") < 40, "18-39")
        .when(F.col("age") < 60, "40-59")
        .otherwise("60+"),
    )
    return patients.cache()


def make_summary(patients):
    """Count records and summarize risk and missing measurements by group."""
    # For missing-value columns, 1 means missing and 0 means present.
    return (
        patients.groupBy("source_dataset", "sex", "age_band")
        .agg(
            F.count("*").alias("n"),
            F.round(F.avg("risk_label"), 3).alias("risk_rate"),
            F.round(F.avg(F.col("bmi").isNull().cast("int")), 3).alias(
                "bmi_missing"
            ),
            F.round(
                F.avg(F.col("glucose_mgdl").isNull().cast("int")), 3
            ).alias("glucose_missing"),
        )
        .orderBy("source_dataset", "sex", "age_band")
    )


def load_who_data():
    """Read country-level WHO indicator files and use consistent column names."""
    frames = []
    for filename in glob.glob(f"{RAW}/who_*.json"):
        with open(filename, encoding="utf-8") as who_file:
            data = pd.DataFrame(json.load(who_file))

        columns = [
            "IndicatorCode",
            "SpatialDim",
            "SpatialDimType",
            "TimeDim",
            "Dim1",
            "NumericValue",
        ]
        data = data.reindex(columns=columns)
        # The app shows country-level context, not regional WHO records.
        data = data[data["SpatialDimType"] == "COUNTRY"]
        data = data.rename(
            columns={
                "IndicatorCode": "indicator_code",
                "SpatialDim": "country",
                "TimeDim": "year",
                "Dim1": "dim1",
                "NumericValue": "value",
            }
        )
        frames.append(data.drop(columns="SpatialDimType"))

    return pd.concat(frames, ignore_index=True)


def load_provenance():
    """Read the download log and add processing information."""
    provenance = pd.read_json(f"{RAW}/provenance_log.jsonl", lines=True)
    provenance["transform_version"] = TRANSFORM_VERSION
    provenance["processed_at"] = pd.Timestamp.utcnow().isoformat()
    return provenance


def save_tables(patients, summary, who, provenance):
    """Save all integrated data tables to the SQLite database."""
    with sqlite3.connect(DB_PATH) as connection:
        # Replace old tables so the database reflects this pipeline run.
        patients.to_sql("patient_record", connection, if_exists="replace", index=False)
        summary.to_sql("risk_summary", connection, if_exists="replace", index=False)
        who.to_sql("country_indicator", connection, if_exists="replace", index=False)
        provenance.to_sql("provenance", connection, if_exists="replace", index=False)


def main():
    # Start Spark locally; it handles the table joins and summaries.
    spark = (
        SparkSession.builder.appName("predictive-health-radar")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Prepare each source, make one patient table, and calculate summaries.
        sources = [
            load_nhanes(spark),
            load_heart_data(spark),
            load_pima_data(spark),
        ]
        patients_spark = combine_patient_data(sources)
        summary_spark = make_summary(patients_spark)

        patients = patients_spark.toPandas()
        summary = summary_spark.toPandas()
    finally:
        # Always stop Spark, even if a processing step raises an error.
        spark.stop()

    # pandas handles the smaller WHO files and writes the final SQLite tables.
    who = load_who_data()
    provenance = load_provenance()
    save_tables(patients, summary, who, provenance)

    print(patients.groupby("source_dataset").size())
    print(summary.head(12))


if __name__ == "__main__":
    main()
