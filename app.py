"""Step 5 - Streamlit app.  Run:  streamlit run src/app.py"""
import sqlite3

import chromadb
import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from common import CHROMA_PATH, DB_PATH, describe

st.set_page_config(page_title="Predictive Health Radar")
st.title("Predictive Health Radar")
st.caption("Educational early-warning demo for metabolic/cardiovascular risk. Not medical advice.")


@st.cache_resource
def load():
    # Load these large resources once instead of reloading on every app update.
    bundle = joblib.load("data/model.joblib")
    coll = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("patients")
    return bundle, coll, SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


bundle, coll, embedder = load()

# Collect measurements from the user.
c1, c2 = st.columns(2)
age = c1.number_input("Age", 18, 100, 50)
sex = c2.selectbox("Sex", ["F", "M"])
bmi = c1.number_input("BMI", 12.0, 60.0, 27.0)
sbp = c2.number_input("Systolic BP (mmHg)", 80, 220, 125)
dbp = c1.number_input("Diastolic BP (mmHg)", 40, 140, 80)
glu = c2.number_input("Fasting glucose (mg/dL)", 50, 400, 95)
chol = c1.number_input("Total cholesterol (mg/dL)", 80, 400, 190)
smoker = c2.selectbox("Ever smoked 100+ cigarettes?", ["No", "Yes"])

if st.button("Assess risk"):
    # Make the input row match the model's training columns.
    row = {"age": age, "sex": sex, "bmi": bmi, "systolic_bp": sbp, "diastolic_bp": dbp,
           "glucose_mgdl": glu, "cholesterol_mgdl": chol, "smoker": int(smoker == "Yes")}
    X = pd.DataFrame([{**row, "sex_m": float(sex == "M")}])[bundle["features"]]
    # Predict a probability, then display a simple risk category.
    p = float(bundle["model"].predict_proba(X)[0, 1])
    tier, advice = (("Low", "Keep up healthy habits and routine checkups.") if p < 0.2 else
                    ("Moderate", "Consider scheduling a checkup to discuss these numbers.") if p < 0.5 else
                    ("High", "Consider talking to a clinician soon. Seek urgent care for chest pain or severe symptoms."))
    st.subheader(f"{tier} risk ({p:.0%})")
    st.write(advice)

    # Search for patient descriptions that are close to this user's profile.
    res = coll.query(query_embeddings=embedder.encode([describe(row)]).tolist(), n_results=5)
    st.markdown("**Similar profiles in the integrated data**")
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        st.write(f"- {doc}  _(source: {meta['source']}, at-risk label: {meta['risk_label']})_")

    # Show recent US WHO indicators as population context.
    try:
        with sqlite3.connect(DB_PATH) as con:
            who = pd.read_sql("SELECT indicator_code, year, value FROM country_indicator "
                              "WHERE country='USA' AND (dim1 IS NULL OR dim1='SEX_BTSX') "
                              "ORDER BY year DESC", con).drop_duplicates("indicator_code")
        st.markdown("**US population context (WHO GHO)**")
        st.dataframe(who)
    except Exception:
        pass
