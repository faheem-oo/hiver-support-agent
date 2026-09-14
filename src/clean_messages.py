import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/applesupport_threads.csv"
)

OUTPUT_PATH = Path(
    "data/processed/applesupport_customer_clean.csv"
)


def clean_text(text):
    """Remove Twitter-specific noise while preserving meaning."""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove HTML entities
    text = re.sub(
        r"&\w+;",
        " ",
        text
    )

    # Remove hashtags symbol but keep the word
    text = re.sub(
        r"#",
        "",
        text
    )

    # Replace numbers
    text = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


print("Loading conversation data...")

df = pd.read_csv(INPUT_PATH)

# Customer messages only
customer = df[
    df["inbound"] == True
].copy()

customer["clean_text"] = (
    customer["text"]
    .fillna("")
    .apply(clean_text)
)

# Remove very short messages
customer = customer[
    customer["clean_text"].str.len() >= 20
].copy()

print(
    f"Customer messages after cleaning: "
    f"{len(customer):,}"
)

print("\n" + "=" * 60)
print("BEFORE → AFTER")
print("=" * 60)

for _, row in customer.head(20).iterrows():

    print("\nBEFORE:")
    print(row["text"])

    print("\nAFTER:")
    print(row["clean_text"])

# Save
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

customer.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(OUTPUT_PATH)