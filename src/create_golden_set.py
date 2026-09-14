from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/applesupport_training_messages.csv")
OUTPUT = Path("data/processed/golden_set.csv")

SAMPLE_SIZE = 200


def main():
    if not INPUT.exists():
        print(f"ERROR: File not found: {INPUT}")
        return

    df = pd.read_csv(INPUT)

    print(f"Available messages: {len(df):,}")

    # Fixed random seed makes the sample reproducible
    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42
    ).copy()

    # Create annotation columns
    sample.insert(0, "id", range(1, len(sample) + 1))

    sample["intent"] = ""
    sample["escalate"] = ""
    sample["escalation_reason"] = ""

    # Keep only the fields needed for annotation
    columns = [
        "id",
        "tweet_id",
        "text",
        "clean_text",
        "intent",
        "escalate",
        "escalation_reason"
    ]

    sample = sample[columns]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT, index=False)

    print(f"\nCreated golden set: {OUTPUT}")
    print(f"Examples: {len(sample)}")

    print("\nIntent labels to use:")
    print("1. ios_update")
    print("2. battery_charging")
    print("3. device_performance")
    print("4. screen_display")
    print("5. connectivity")
    print("6. apple_id_icloud")
    print("7. apps")
    print("8. itunes_music")
    print("9. app_store_purchases")
    print("10. apple_watch")
    print("11. other")

    print("\nEscalate:")
    print("yes / no")


if __name__ == "__main__":
    main()