import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Allow importing agent.py from src/
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from agent import run_agent


GOLDEN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_set.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "agent_eval.csv"
)


def main():

    golden = pd.read_csv(GOLDEN_PATH)

    print(f"Golden examples: {len(golden)}")
    print("Running agent evaluation...")
    print()

    results = []

    for i, row in golden.iterrows():

        message = str(row["text"])

        result = run_agent(message)

        results.append(
            {
                "id": row["id"],
                "tweet_id": row["tweet_id"],
                "text": message,

                "actual_intent":
                    row["intent"],

                "predicted_intent":
                    result["intent"],

                "intent_confidence":
                    result["confidence"],

                "actual_escalate":
                    row["escalate"],

                "predicted_escalate":
                    "yes"
                    if result["escalate"]
                    else "no",

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

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{len(golden)}"
            )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -------------------------------------------------
    # Intent evaluation
    # -------------------------------------------------

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
    print("INTENT RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: {intent_accuracy:.4f}"
    )

    print(
        f"Macro F1: {intent_macro_f1:.4f}"
    )

    print()
    print(
        classification_report(
            results_df["actual_intent"],
            results_df["predicted_intent"],
            zero_division=0
        )
    )

    # -------------------------------------------------
    # Escalation evaluation
    # -------------------------------------------------

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

    escalation_f1_yes = f1_score(
        results_df["actual_escalate"],
        results_df["predicted_escalate"],
        pos_label="yes",
        zero_division=0
    )

    print("=" * 60)
    print("ESCALATION RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: {escalation_accuracy:.4f}"
    )

    print(
        f"Macro F1: {escalation_macro_f1:.4f}"
    )

    print(
        f"Escalation F1: {escalation_f1_yes:.4f}"
    )

    # -------------------------------------------------
    # Error analysis
    # -------------------------------------------------

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
    print("ERROR COUNTS")
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
    print("FIRST 20 INTENT ERRORS")
    print("=" * 60)

    for _, row in intent_errors.head(20).iterrows():

        print()
        print(f"ID: {row['id']}")

        print(
            f"Actual: "
            f"{row['actual_intent']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_intent']}"
        )

        print(
            f"Message: "
            f"{row['text']}"
        )

    print()
    print("=" * 60)
    print("FIRST 20 ESCALATION ERRORS")
    print("=" * 60)

    for _, row in escalation_errors.head(20).iterrows():

        print()
        print(f"ID: {row['id']}")

        print(
            f"Actual: "
            f"{row['actual_escalate']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_escalate']}"
        )

        print(
            f"Message: "
            f"{row['text']}"
        )

    print()
    print("=" * 60)

    print(
        f"Saved evaluation to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()