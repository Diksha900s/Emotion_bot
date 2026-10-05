"""Train TF-IDF + LinearSVC emotion classifier using the bundled dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "emotions.csv"
MODEL_PATH = PROJECT_ROOT / "model.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_cols = {"text", "emotion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["clean"] = df["text"].map(clean)
    Xtr, Xte, ytr, yte = train_test_split(
        df["clean"],
        df["emotion"],
        test_size=0.25,
        stratify=df["emotion"],
        random_state=42,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("svm", LinearSVC(C=1.0)),
    ])
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print(f"Test accuracy: {accuracy_score(yte, pred):.2f}\n")
    print(classification_report(yte, pred, zero_division=0))

    model.fit(df["clean"], df["emotion"])  # refit on all data for the chatbot
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
