import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

DATA_PATH = Path(
    "data/processed/applesupport_threads.csv"
)

OUTPUT_PATH = Path(
    "outputs/intent_clusters.csv"
)

print("Loading AppleSupport threads...")

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------
# Keep customer messages only
# ---------------------------------------------------------

customer = df[df["inbound"] == True].copy()

customer["text"] = (
    customer["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove extremely short messages
customer = customer[
    customer["text"].str.len() >= 15
].copy()

print(
    f"Customer messages available: "
    f"{len(customer):,}"
)

# ---------------------------------------------------------
# Sample messages
#
# We don't need all 100k+ messages for clustering.
# ---------------------------------------------------------

sample_size = min(30000, len(customer))

sample = customer.sample(
    n=sample_size,
    random_state=42
).copy()

print(
    f"Using {len(sample):,} messages for clustering."
)

# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5
)

X = vectorizer.fit_transform(sample["text"])

print(
    f"TF-IDF matrix: {X.shape}"
)

# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------

N_CLUSTERS = 15

print(
    f"\nCreating {N_CLUSTERS} clusters..."
)

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=1024,
    n_init=10
)

sample["cluster"] = model.fit_predict(X)

# ---------------------------------------------------------
# Find important words for each cluster
# ---------------------------------------------------------

terms = vectorizer.get_feature_names_out()

print("\n" + "=" * 70)
print("INTENT CLUSTERS")
print("=" * 70)

cluster_info = []

for cluster_id in range(N_CLUSTERS):

    cluster_indices = (
        sample["cluster"].to_numpy() == cluster_id
    )

    cluster_matrix = X[cluster_indices]

    # Average TF-IDF score for each word
    mean_scores = cluster_matrix.mean(axis=0)
    mean_scores = mean_scores.A1

    top_indices = mean_scores.argsort()[-15:][::-1]

    top_words = [
        terms[i]
        for i in top_indices
    ]

    count = cluster_indices.sum()

    print(
        f"\nCLUSTER {cluster_id} "
        f"({count:,} messages)"
    )

    print(
        "Keywords:",
        ", ".join(top_words)
    )

    # Show example messages
    examples = sample[
        cluster_indices
    ]["text"].head(5)

    print("\nExamples:")

    for example in examples:
        print(
            f"  - {example[:250]}"
        )

    cluster_info.append({
        "cluster": cluster_id,
        "count": count,
        "keywords": ", ".join(top_words)
    })

# ---------------------------------------------------------
# Save cluster information
# ---------------------------------------------------------

cluster_df = pd.DataFrame(cluster_info)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

cluster_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_PATH)