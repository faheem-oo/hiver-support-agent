import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


INPUT_FILE = "data/processed/weak_labels.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_FILE = MODEL_DIR / "intent_tfidf.joblib"
MODEL_FILE = MODEL_DIR / "intent_classifier.joblib"


def main():
    print("Loading weak-labeled data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Examples: {len(df):,}")

    X_text = df["clean_text"].fillna("")
    y = df["weak_intent"]

    print("\n=== Training distribution ===")
    print(y.value_counts().sort_index().to_string())

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(X_text)

    print(f"Feature matrix: {X.shape}")

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X, y)

    print("\nSaving model...")

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(VECTORIZER_FILE)
    print(MODEL_FILE)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()