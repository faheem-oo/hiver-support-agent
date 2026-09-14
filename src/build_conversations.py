import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/archive/twcs/twcs.csv")
OUTPUT_PATH = Path("data/processed/applesupport_conversations.csv")

BRAND = "AppleSupport"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df):,} tweets")

# ---------------------------------------------------------
# 1. Get AppleSupport tweets
# ---------------------------------------------------------

brand = df[df["author_id"] == BRAND].copy()

print(f"AppleSupport tweets: {len(brand):,}")

# ---------------------------------------------------------
# 2. Create tweet lookup
# ---------------------------------------------------------

tweet_lookup = df.set_index("tweet_id")

# ---------------------------------------------------------
# 3. Find customer tweets that AppleSupport directly replied to
# ---------------------------------------------------------

brand_parent_ids = set(
    brand["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
)

customer_messages = df[
    (df["inbound"] == True) &
    (df["tweet_id"].isin(brand_parent_ids))
].copy()

print(
    f"Customer messages with direct AppleSupport response: "
    f"{len(customer_messages):,}"
)

# ---------------------------------------------------------
# 4. Build customer → AppleSupport pairs
# ---------------------------------------------------------

rows = []

for _, customer in customer_messages.iterrows():

    customer_id = int(customer["tweet_id"])

    # AppleSupport's response to this customer
    responses = brand[
        brand["in_response_to_tweet_id"] == customer_id
    ]

    if len(responses) == 0:
        continue

    # Usually there should be one direct response
    for _, response in responses.iterrows():

        rows.append({
            "customer_tweet_id": customer_id,
            "customer_text": customer["text"],
            "customer_created_at": customer["created_at"],

            "agent_tweet_id": int(response["tweet_id"]),
            "agent_text": response["text"],
            "agent_created_at": response["created_at"]
        })

conversations = pd.DataFrame(rows)

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

conversations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CONVERSATION DATASET")
print("=" * 60)

print(f"Customer → AppleSupport pairs: {len(conversations):,}")

print("\nColumns:")
for column in conversations.columns:
    print("-", column)

print("\n" + "=" * 60)
print("SAMPLE CONVERSATIONS")
print("=" * 60)

for i, row in conversations.head(15).iterrows():

    print(f"\n--- Conversation {i + 1} ---")

    print("CUSTOMER:")
    print(row["customer_text"])

    print("\nAPPLESUPPORT:")
    print(row["agent_text"])

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(OUTPUT_PATH)