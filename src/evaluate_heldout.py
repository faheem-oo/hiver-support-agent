import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from agent import run_agent


TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_test.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "heldout_results.csv"
)


def main():

    test = pd.read_csv(TEST_PATH)

    print(f"Held-out test examples: {len(test)}")
    print("Running final evaluation...")
    print()

    results = []

    for i, row in test.iterrows():

        result = run_agent(str(row["text"]))

        results.append(
            {
                "id": row["id"],
                "tweet_id": row["tweet_id"],
                "text": row["text"],

                "actual_intent":
                    row["intent"],

                "predicted_intent":
                    result["intent"],

                "intent_confidence":
                    result["confidence"],

                "actual_escalate":
                    row["escalate"],

                "predicted_escalate":
                    "yes" if result["escalate"] else "no",

                "escalation_reason":
                    result["escalation_reason"],

                "actual_escalation_reason":
                    row["escalation_reason"],

                "retrieval_score":
                    result["retrieval_score"],

                "draft_reply":
                    result["reply"],
            }
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test)}")

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # Intent
    # --------------------------------------------------

    intent_accuracy = accuracy_score(
        results_df["actual_intent"],
        results_df["predicted_intent"]
    )

    intent_macro_f1 = f1_score(
        results_df["actual_intent"],
        results_df["predicted_intent"],
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 60)
    print("HELD-OUT INTENT RESULTS")
    print("=" * 60)

    print(f"Accuracy: {intent_accuracy:.4f}")
    print(f"Macro F1: {intent_macro_f1:.4f}")

    print()
    print(
        classification_report(
            results_df["actual_intent"],
            results_df["predicted_intent"],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Escalation
    # --------------------------------------------------

    escalation_accuracy = accuracy_score(
        results_df["actual_escalate"],
        results_df["predicted_escalate"]
    )

    escalation_macro_f1 = f1_score(
        results_df["actual_escalate"],
        results_df["predicted_escalate"],
        average="macro",
        zero_division=0
    )

    escalation_f1 = f1_score(
        results_df["actual_escalate"],
        results_df["predicted_escalate"],
        pos_label="yes",
        zero_division=0
    )

    print("=" * 60)
    print("HELD-OUT ESCALATION RESULTS")
    print("=" * 60)

    print(f"Accuracy: {escalation_accuracy:.4f}")
    print(f"Macro F1: {escalation_macro_f1:.4f}")
    print(f"Escalation F1: {escalation_f1:.4f}")

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    intent_errors = results_df[
        results_df["actual_intent"]
        != results_df["predicted_intent"]
    ]

    escalation_errors = results_df[
        results_df["actual_escalate"]
        != results_df["predicted_escalate"]
    ]

    print()
    print("=" * 60)
    print("HELD-OUT ERROR COUNTS")
    print("=" * 60)

    print(
        f"Intent errors: "
        f"{len(intent_errors)} / {len(results_df)}"
    )

    print(
        f"Escalation errors: "
        f"{len(escalation_errors)} / {len(results_df)}"
    )

    print()
    print("=" * 60)
    print("HELD-OUT INTENT ERRORS")
    print("=" * 60)

    for _, row in intent_errors.iterrows():

        print()
        print(f"ID: {row['id']}")
        print(f"Actual: {row['actual_intent']}")
        print(f"Predicted: {row['predicted_intent']}")
        print(f"Message: {row['text']}")

    print()
    print("=" * 60)
    print("HELD-OUT ESCALATION ERRORS")
    print("=" * 60)

    for _, row in escalation_errors.iterrows():

        print()
        print(f"ID: {row['id']}")
        print(f"Actual: {row['actual_escalate']}")
        print(f"Predicted: {row['predicted_escalate']}")
        print(f"Message: {row['text']}")

    print()
    print("=" * 60)
    print("ESCALATION CONFUSION MATRIX")
    print("=" * 60)

    print(
        confusion_matrix(
            results_df["actual_escalate"],
            results_df["predicted_escalate"],
            labels=["no", "yes"]
        )
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()