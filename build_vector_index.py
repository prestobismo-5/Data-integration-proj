"""Create a Chroma search index from the patient records.

Run from the project folder: python src/build_vector_index.py
"""
import sqlite3

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from common import CHROMA_PATH, DB_PATH, SEED, describe

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RECORDS_PER_SOURCE = 2000
BATCH_SIZE = 2000


def load_patient_data():
    """Read labeled records from SQLite and sample each data source."""
    with sqlite3.connect(DB_PATH) as connection:
        patients = pd.read_sql(
            "SELECT * FROM patient_record WHERE risk_label IS NOT NULL",
            connection,
        )

    sampled_sources = []
    for _, source_data in patients.groupby("source_dataset"):
        # Use the same maximum for each source so one source does not dominate.
        sample_size = min(len(source_data), MAX_RECORDS_PER_SOURCE)
        sampled_sources.append(
            source_data.sample(n=sample_size, random_state=SEED)
        )

    return pd.concat(sampled_sources, ignore_index=True)


def make_documents(patients):
    """Turn each patient record into text and matching Chroma metadata."""
    documents = []
    metadata = []

    for record in patients.to_dict("records"):
        documents.append(describe(record))
        # Keep details needed to identify each search result.
        metadata.append({
            "record_id": int(record["record_id"]),
            "source": record["source_dataset"],
            "risk_label": int(record["risk_label"]),
        })

    return documents, metadata


def clear_collection(collection):
    """Remove old records so running this script rebuilds the index."""
    # Chroma accepts deletes in batches, so clear the collection a piece at a time.
    old_records = collection.get()
    old_ids = old_records["ids"]

    for start in range(0, len(old_ids), BATCH_SIZE):
        collection.delete(ids=old_ids[start:start + BATCH_SIZE])


def save_in_batches(collection, ids, vectors, documents, metadata):
    """Add the patient profiles in batches to Chroma."""
    # Smaller batches avoid sending every profile in one large request.
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadata[start:end],
        )


def main():
    patients = load_patient_data()
    documents, metadata = make_documents(patients)

    # Convert each text description into numbers that can be compared by similarity.
    embedder = SentenceTransformer(MODEL_NAME)
    vectors = embedder.encode(
        documents,
        show_progress_bar=True,
        batch_size=64,
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="patients",
        metadata={"hnsw:space": "cosine"},
    )
    # Rebuild the index so it matches the current patient data.
    clear_collection(collection)

    ids = [str(record_id) for record_id in patients["record_id"]]
    save_in_batches(collection, ids, vectors, documents, metadata)

    print("Indexed", collection.count(), "patient profiles")


if __name__ == "__main__":
    main()
