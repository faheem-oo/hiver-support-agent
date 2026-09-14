import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/archive/twcs/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", len(df), "tweets")

# Brand/support accounts are usually authors of OUTBOUND tweets.
# inbound=False means the tweet was sent by the company/support account.
outbound = df[df["inbound"] == False]

print("\nTotal outbound/support tweets:", len(outbound))

# Count tweets by author
brand_counts = outbound["author_id"].value_counts()

print("\n" + "=" * 60)
print("TOP SUPPORT ACCOUNTS")
print("=" * 60)

print(brand_counts.head(50).to_string())

# Save results
output_path = Path("outputs/brand_counts.csv")
output_path.parent.mkdir(exist_ok=True)

brand_counts.to_csv(
    output_path,
    header=["tweet_count"]
)

print("\nSaved to:", output_path)