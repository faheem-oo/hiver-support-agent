import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_set.csv"
)

DEV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_dev.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_test.csv"
)


def main():

    df = pd.read_csv(INPUT_PATH)

    # Fixed seed makes the split reproducible.
    shuffled = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    dev = shuffled.iloc[:150].copy()
    test = shuffled.iloc[150:].copy()

    dev.to_csv(
        DEV_PATH,
        index=False
    )

    test.to_csv(
        TEST_PATH,
        index=False
    )

    print(f"Total examples: {len(df)}")
    print(f"Development set: {len(dev)}")
    print(f"Held-out test set: {len(test)}")

    print()
    print("Development escalation distribution:")
    print(dev["escalate"].value_counts())

    print()
    print("Held-out test escalation distribution:")
    print(test["escalate"].value_counts())

    print()
    print(f"Saved: {DEV_PATH}")
    print(f"Saved: {TEST_PATH}")


if __name__ == "__main__":
    main()