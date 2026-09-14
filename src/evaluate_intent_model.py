import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


GOLDEN_FILE = "data/processed/golden_set.csv"

VECTORIZER_FILE = "models/intent_tfidf.joblib"
MODEL_FILE = "models/intent_classifier.joblib"

OUTPUT_FILE = "outputs/intent_model_results.csv"


def main():
    print("Loading golden set...")

    df = pd.read_csv(GOLDEN_FILE)

    print(f"Golden examples: {len(df)}")

    # Only evaluate examples that have a human intent label.
    df = df[df["intent"].notna()].copy()

    X_text = df["clean_text"].fillna("")
    y_true = df["intent"]

    print("\nLoading trained model...")

    vectorizer = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)

    print("Generating predictions...")

    X = vectorizer.transform(X_text)

    y_pred = model.predict(X)

    df["predicted_intent"] = y_pred

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n=== Intent Model Results ===")

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n=== Classification Report ===")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    print("\n=== Confusion Matrix ===")

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(cm_df.to_string())

    # Save predictions for failure analysis.
    output = df[
        [
            "id",
            "tweet_id",
            "text",
            "clean_text",
            "intent",
            "predicted_intent",
        ]
    ].copy()

    output["correct"] = (
        output["intent"]
        == output["predicted_intent"]
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSaved predictions to:")
    print(OUTPUT_FILE)

    print("\n=== Incorrect Predictions ===")

    errors = output[
        ~output["correct"]
    ]

    print(
        f"Errors: {len(errors)} / {len(output)}"
    )

    for _, row in errors.head(20).iterrows():
        print("\n---")
        print(f"ID:       {row['id']}")
        print(f"Text:     {row['text']}")
        print(f"Actual:   {row['intent']}")
        print(f"Predicted:{row['predicted_intent']}")


if __name__ == "__main__":
    main()