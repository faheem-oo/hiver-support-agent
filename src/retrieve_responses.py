import re
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/applesupport_retrieval.csv"

VECTORIZER_FILE = "models/retrieval_tfidf.joblib"
MATRIX_FILE = "models/retrieval_matrix.joblib"


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_index():
    print("Loading retrieval data...")

    df = pd.read_csv(INPUT_FILE)

    # Only use useful historical responses.
    df = df[
        df["response_is_generic"] == False
    ].copy()

    print(
        f"Useful historical examples: {len(df):,}"
    )

    df["retrieval_text"] = (
        df["customer_text"]
        .fillna("")
        .apply(clean_text)
    )

    print("\nBuilding TF-IDF index...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        df["retrieval_text"]
    )

    print(
        f"Retrieval matrix: {matrix.shape}"
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    joblib.dump(
        matrix,
        MATRIX_FILE
    )

    print("\nSaved:")
    print(VECTORIZER_FILE)
    print(MATRIX_FILE)

    return df, vectorizer, matrix


def retrieve(
    query,
    df,
    vectorizer,
    matrix,
    top_k=5,
):
    query_clean = clean_text(query)

    query_vector = vectorizer.transform(
        [query_clean]
    )

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "score": float(scores[index]),
                "customer_text": df.iloc[index][
                    "customer_text"
                ],
                "response_text": df.iloc[index][
                    "response_text"
                ],
            }
        )

    return results


def main():
    df, vectorizer, matrix = build_index()

    print("\n=== Retrieval test ===")

    test_queries = [
        "My iPhone battery is draining very quickly after the iOS update",
        "My iPhone won't connect to WiFi",
        "My Apple ID password is not working",
        "My iPhone screen is not responding to touch",
        "Apple Music won't play my songs",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print("QUERY:")
        print(query)

        results = retrieve(
            query,
            df,
            vectorizer,
            matrix,
            top_k=3,
        )

        for i, result in enumerate(
            results,
            start=1
        ):
            print(
                f"\n--- Result {i} "
                f"(similarity={result['score']:.3f}) ---"
            )

            print("Historical customer:")
            print(result["customer_text"])

            print("\nHistorical AppleSupport response:")
            print(result["response_text"])


if __name__ == "__main__":
    main()