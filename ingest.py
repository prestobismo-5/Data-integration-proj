"""Step 1 - Ingest raw data from 4 sources (files, ARFF/CSV, REST API) and log provenance.
Run from the project root:  python src/ingest.py
"""
import datetime as dt
import hashlib
import io
import json
import pathlib

import pandas as pd
import requests

# Make the folders for downloaded files and their source log.
RAW = pathlib.Path("data/raw")
(RAW / "nhanes").mkdir(parents=True, exist_ok=True)
PROV = RAW / "provenance_log.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 (CS3312 student project)"}

# NHANES 2017-2018 cycle (suffix _J). Verify file names at
# https://wwwn.cdc.gov/nchs/nhanes/ if a download fails.
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
NHANES_FILES = ["DEMO_J", "BMX_J", "BPX_J", "GLU_J", "TCHOL_J", "SMQ_J", "DIQ_J", "BPQ_J"]
PIMA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "pima-indians-diabetes.data.csv")
PIMA_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
             "insulin", "bmi", "dpf", "age", "outcome"]
WHO_BASE = "https://ghoapi.azureedge.net/api/"
# Check codes with: https://ghoapi.azureedge.net/api/Indicator?$filter=contains(IndicatorName,'obesity')
WHO_INDICATORS = ["NCD_BMI_30A", "NCD_GLUC_04", "NCD_HYP_PREVALENCE_A"]


def sha256(path):
    # A hash can be used to check whether the downloaded file has changed.
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def log(source, fmt, url, path, rows):
    # Keep a record of where each file came from and when it was downloaded.
    rec = {"source": source, "format": fmt, "url": url, "file": str(path),
           "rows": int(rows), "sha256": sha256(path),
           "retrieved_at": dt.datetime.utcnow().isoformat() + "Z"}
    with PROV.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"ok  {source:<22} {rows:>7} rows")


def fetch(url):
    # Download a file and report an error if the server responds unsuccessfully.
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.content


def main():
    # Start a new download log for this run.
    PROV.unlink(missing_ok=True)

    # 1) CDC NHANES (SAS XPT -> CSV)
    for name in NHANES_FILES:
        url = f"{NHANES_BASE}{name}.xpt"
        xpt = RAW / "nhanes" / f"{name}.xpt"
        xpt.write_bytes(fetch(url))
        df = pd.read_sas(xpt, format="xport", encoding="utf-8")
        csv = RAW / "nhanes" / f"{name}.csv"
        df.to_csv(csv, index=False)
        log(f"NHANES {name}", "XPT->CSV", url, csv, len(df))

    # 2) UCI Heart Disease (Cleveland) via ucimlrepo
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    # Put the input columns and target column together in one table.
    heart = pd.concat([ds.data.features, ds.data.targets], axis=1)
    p = RAW / "uci_heart.csv"
    heart.to_csv(p, index=False)
    log("UCI Heart Disease", "CSV", "https://archive.ics.uci.edu/dataset/45", p, len(heart))

    # 3) Pima Indians Diabetes (public CSV mirror; swap for a Kaggle download if preferred)
    # The source file has no header, so provide column names while reading it.
    pima = pd.read_csv(io.StringIO(fetch(PIMA_URL).decode()), header=None, names=PIMA_COLS)
    p = RAW / "pima_diabetes.csv"
    pima.to_csv(p, index=False)
    log("Pima Diabetes", "CSV", PIMA_URL, p, len(pima))

    # 4) WHO Global Health Observatory (REST/OData JSON)
    # Save the selected country indicators as separate JSON files.
    for ind in WHO_INDICATORS:
        url = WHO_BASE + ind
        data = json.loads(fetch(url))["value"]
        p = RAW / f"who_{ind}.json"
        p.write_text(json.dumps(data))
        log(f"WHO {ind}", "JSON API", url, p, len(data))


if __name__ == "__main__":
    main()
