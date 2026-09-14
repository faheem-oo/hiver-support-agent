import pandas as pd

PATH = "data/processed/golden_set.csv"

EXPECTED_INTENTS = {
    "ios_update",
    "battery_charging",
    "device_performance",
    "screen_display",
    "connectivity",
    "apple_id_icloud",
    "apps",
    "itunes_music",
    "app_store_purchases",
    "apple_watch",
    "other",
}

df = pd.read_csv(PATH, keep_default_na=False)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Check columns
expected_columns = [
    "id",
    "tweet_id",
    "text",
    "clean_text",
    "intent",
    "escalate",
    "escalation_reason",
]

if list(df.columns) != expected_columns:
    print("\nERROR: Column structure is incorrect.")

# Check row count
if len(df) != 200:
    print(f"\nERROR: Expected 200 rows, found {len(df)}")

# Check IDs
if set(df["id"]) != set(range(1, 201)):
    print("\nERROR: IDs are not exactly 1-200.")

# Check duplicate tweet IDs
duplicates = df[df["tweet_id"].duplicated(keep=False)]

if len(duplicates) > 0:
    print("\nWARNING: Duplicate tweet IDs:")
    print(duplicates[["id", "tweet_id"]])

# Check intents
invalid_intents = df[~df["intent"].isin(EXPECTED_INTENTS)]

if len(invalid_intents) > 0:
    print("\nERROR: Invalid intents:")
    print(invalid_intents[["id", "intent"]])

# Check escalation values
invalid_escalation = df[~df["escalate"].isin(["yes", "no"])]

if len(invalid_escalation) > 0:
    print("\nERROR: Invalid escalation values:")
    print(invalid_escalation[["id", "escalate"]])

# Check escalation reasons
missing_reason = df[
    (df["escalate"] == "yes") &
    (df["escalation_reason"].str.strip() == "")
]

if len(missing_reason) > 0:
    print("\nERROR: Escalated examples without a reason:")
    print(missing_reason[["id", "intent", "escalation_reason"]])

# Check unexpected reasons on non-escalated examples
unexpected_reason = df[
    (df["escalate"] == "no") &
    (df["escalation_reason"].str.strip() != "")
]

if len(unexpected_reason) > 0:
    print("\nWARNING: Non-escalated examples have escalation reasons:")
    print(unexpected_reason[["id", "escalation_reason"]])

print("\n=== Intent distribution ===")
print(df["intent"].value_counts())

print("\n=== Escalation distribution ===")
print(df["escalate"].value_counts())

print("\n=== Escalation by intent ===")
print(pd.crosstab(df["intent"], df["escalate"]))

print("\nValidation complete.")