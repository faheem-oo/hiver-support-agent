import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/archive/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/applesupport_threads.csv")

BRAND = "AppleSupport"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df):,} tweets")

# ---------------------------------------------------------
# Prepare IDs
# ---------------------------------------------------------

df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
df["in_response_to_tweet_id"] = pd.to_numeric(
    df["in_response_to_tweet_id"],
    errors="coerce"
)

df = df.dropna(subset=["tweet_id"])
df["tweet_id"] = df["tweet_id"].astype(int)

# ---------------------------------------------------------
# Create tweet lookup
# ---------------------------------------------------------

tweet_lookup = df.set_index("tweet_id").to_dict("index")

print("Tweet lookup created.")

# ---------------------------------------------------------
# Find AppleSupport tweets
# ---------------------------------------------------------

brand_tweets = df[df["author_id"] == BRAND].copy()

brand_ids = set(brand_tweets["tweet_id"])

print(f"AppleSupport tweets: {len(brand_tweets):,}")

# ---------------------------------------------------------
# Find conversation root
# ---------------------------------------------------------

def find_root(tweet_id, max_depth=50):

    current = tweet_id
    visited = set()

    for _ in range(max_depth):

        if current in visited:
            break

        visited.add(current)

        tweet = tweet_lookup.get(current)

        if tweet is None:
            break

        parent = tweet["in_response_to_tweet_id"]

        if pd.isna(parent):
            break

        current = int(parent)

    return current


print("Finding conversation roots...")

roots = set()

for tweet_id in brand_ids:
    roots.add(find_root(tweet_id))

print(f"Unique conversation roots: {len(roots):,}")

# ---------------------------------------------------------
# Build parent → children mapping
# ---------------------------------------------------------

print("Building conversation tree...")

children = {}

for _, row in df.iterrows():

    parent = row["in_response_to_tweet_id"]

    if pd.isna(parent):
        continue

    parent = int(parent)
    tweet_id = int(row["tweet_id"])

    children.setdefault(parent, []).append(tweet_id)

print("Conversation tree created.")

# ---------------------------------------------------------
# Reconstruct conversations
# ---------------------------------------------------------

conversation_rows = []

for conversation_number, root in enumerate(roots, start=1):

    queue = [root]
    visited = set()
    turns = []

    while queue:

        tweet_id = queue.pop(0)

        if tweet_id in visited:
            continue

        visited.add(tweet_id)

        tweet = tweet_lookup.get(tweet_id)

        if tweet is None:
            continue

        # IMPORTANT:
        # Add tweet_id back into dictionary
        tweet_with_id = {
            "tweet_id": tweet_id,
            **tweet
        }

        turns.append(tweet_with_id)

        for child in children.get(tweet_id, []):
            queue.append(child)

    # Keep only conversations involving AppleSupport
    if not any(
        tweet["author_id"] == BRAND
        for tweet in turns
    ):
        continue

    # Sort chronologically
    turns.sort(key=lambda x: x["created_at"])

    for turn_number, tweet in enumerate(turns, start=1):

        conversation_rows.append({

            "conversation_id": conversation_number,

            "turn_number": turn_number,

            "tweet_id": tweet["tweet_id"],

            "author_id": tweet["author_id"],

            "inbound": tweet["inbound"],

            "created_at": tweet["created_at"],

            "text": tweet["text"]
        })


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

threads = pd.DataFrame(conversation_rows)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

threads.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("THREAD ANALYSIS")
print("=" * 60)

print(
    f"Total conversation turns: "
    f"{len(threads):,}"
)

print(
    f"Unique conversations: "
    f"{threads['conversation_id'].nunique():,}"
)

print("\nTurn distribution:")

turn_counts = (
    threads
    .groupby("conversation_id")
    .size()
    .value_counts()
    .sort_index()
)

print(
    turn_counts.head(20).to_string()
)

# ---------------------------------------------------------
# Sample conversations
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE FULL THREADS")
print("=" * 60)

sample_ids = (
    threads["conversation_id"]
    .drop_duplicates()
    .head(5)
)

for conversation_id in sample_ids:

    conversation = threads[
        threads["conversation_id"] == conversation_id
    ]

    print(
        f"\n--- Conversation {conversation_id} ---"
    )

    for _, row in conversation.iterrows():

        speaker = (
            "CUSTOMER"
            if row["inbound"]
            else "APPLESUPPORT"
        )

        print(f"\n{speaker}:")
        print(row["text"])


print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(OUTPUT_PATH)