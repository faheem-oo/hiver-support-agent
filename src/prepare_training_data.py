from pathlib import Path
import pandas as pd
import re
import html

INPUT = Path("data/processed/applesupport_customer_messages.csv")
OUTPUT = Path("data/processed/applesupport_training_messages.csv")


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities such as &amp;
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol but preserve the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize apostrophes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # Keep letters, numbers, basic punctuation and symbols
    text = re.sub(r"[^\w\s.,!?'+#&/-]", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print(f"Loading: {INPUT}")

    if not INPUT.exists():
        print(f"ERROR: File not found: {INPUT}")
        return

    df = pd.read_csv(INPUT)

    print(f"Original rows: {len(df):,}")

    df["clean_text"] = df["text"].apply(clean_text)

    # Remove empty messages
    df = df[df["clean_text"].str.len() > 0].copy()

    # Remove exact duplicate messages
    df = df.drop_duplicates(subset=["clean_text"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Final rows: {len(df):,}")
    print(f"Saved to: {OUTPUT}")

    print("\nSample cleaned messages:\n")

    for text in df["clean_text"].head(10):
        print("-", text)


if __name__ == "__main__":
    main()