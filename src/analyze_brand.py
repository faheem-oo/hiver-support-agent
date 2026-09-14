import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/archive/twcs/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

BRAND = "AppleSupport"

# All tweets sent by AppleSupport
brand_tweets = df[df["author_id"] == BRAND].copy()

# Customer tweets that mention/reply to AppleSupport
customer_tweets = df[
    (df["inbound"] == True) &
    (df["text"].str.contains(BRAND, case=False, na=False))
].copy()

print("\n" + "=" * 60)
print(f"BRAND ANALYSIS: {BRAND}")
print("=" * 60)

print(f"\nAppleSupport tweets: {len(brand_tweets):,}")
print(f"Customer tweets mentioning AppleSupport: {len(customer_tweets):,}")

# Tweets that AppleSupport replied to
replied_to = brand_tweets["in_response_to_tweet_id"].notna()

print(f"AppleSupport tweets replying to another tweet: {replied_to.sum():,}")

# Customer tweets that received a response
brand_response_ids = set(
    brand_tweets["in_response_to_tweet_id"]
    .dropna()
    .astype(str)
)

customer_tweets["received_brand_response"] = (
    customer_tweets["tweet_id"].astype(str).isin(brand_response_ids)
)

print(
    f"Customer tweets with an AppleSupport response: "
    f"{customer_tweets['received_brand_response'].sum():,}"
)

print("\n" + "=" * 60)
print("SAMPLE CUSTOMER MESSAGES")
print("=" * 60)

for i, text in enumerate(customer_tweets["text"].head(30), 1):
    print(f"\n{i}. {text}")

# Save AppleSupport data
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

brand_tweets.to_csv(
    output_dir / "applesupport_outbound.csv",
    index=False
)

customer_tweets.to_csv(
    output_dir / "applesupport_customer_messages.csv",
    index=False
)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(output_dir / "applesupport_outbound.csv")
print(output_dir / "applesupport_customer_messages.csv")