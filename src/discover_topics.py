import pandas as pd
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = Path(
    "data/processed/applesupport_customer_clean.csv"
)

OUTPUT_PATH = Path(
    "outputs/common_terms.csv"
)

print("Loading customer messages...")

df = pd.read_csv(INPUT_PATH)

texts = (
    df["clean_text"]
    .fillna("")
    .astype(str)
)

# ---------------------------------------------------------
# Words we don't want to count as topics
# ---------------------------------------------------------

STOPWORDS = {
    "apple", "applesupport", "iphone", "phone",
    "help", "please", "just", "can", "cant",
    "can't", "im", "i'm", "ive", "i've",
    "you", "your", "my", "the", "and", "to",
    "a", "an", "is", "it", "this", "that",
    "on", "in", "for", "of", "with", "have",
    "has", "was", "are", "be", "do", "does",
    "did", "not", "but", "so", "now", "what",
    "why", "how", "when", "get", "got",
    "want", "need", "like", "really", "still",
    "one", "also", "would", "could", "thanks",
    "thank", "yes", "no"
}

# ---------------------------------------------------------
# Extract words
# ---------------------------------------------------------

word_counter = Counter()

for text in texts:

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z']+\b",
        text.lower()
    )

    for word in words:

        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        word_counter[word] += 1


# ---------------------------------------------------------
# Show most common terms
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MOST COMMON CUSTOMER TERMS")
print("=" * 60)

for word, count in word_counter.most_common(100):

    print(
        f"{word:<25} {count:>8,}"
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = pd.DataFrame(
    word_counter.most_common(),
    columns=["term", "count"]
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)