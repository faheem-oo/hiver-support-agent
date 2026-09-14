import re
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/applesupport_retrieval.csv"

VECTORIZER_FILE = "models/retrieval_tfidf.joblib"
MATRIX_FILE = "models/retrieval_matrix.joblib"


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def response_quality(text):
    """
    Simple heuristic to distinguish actionable responses
    from responses that mainly redirect the customer to DM.
    """

    text = str(text).lower()

    useful_terms = [
        "try",
        "check",
        "settings",
        "update",
        "restart",
        "reset",
        "verify",
        "steps",
        "article",
        "tap",
        "go to",
        "turn",
        "install",
        "remove",
        "backup",
        "restore",
        "diagnos",
        "version",
        "model",
        "error",
    ]

    score = 0

    for term in useful_terms:
        if term in text:
            score += 1

    # Penalize responses that are essentially DM redirects.
    if "dm us" in text:
        score -= 1

    if "send us a dm" in text:
        score -= 1

    return score


def retrieve(query, df, vectorizer, matrix, top_k=5):
    query_vector = vectorizer.transform(
        [clean_text(query)]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    candidates = []

    for index in similarities.argsort()[::-1][:20]:
        quality = response_quality(
            df.iloc[index]["response_text"]
        )

        candidates.append(
            {
                "similarity": float(similarities[index]),
                "quality": quality,
                "combined_score": (
                    float(similarities[index])
                    + 0.02 * quality
                ),
                "customer_text": df.iloc[index][
                    "customer_text"
                ],
                "response_text": df.iloc[index][
                    "response_text"
                ],
            }
        )

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return candidates[:top_k]


def main():
    print("Loading retrieval data...")

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["response_is_generic"] == False
    ].copy()

    print(
        f"Historical examples: {len(df):,}"
    )

    print("\nLoading retrieval index...")

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    matrix = joblib.load(
        MATRIX_FILE
    )

    queries = [
        "My iPhone battery is draining very quickly after the iOS update",
        "My iPhone won't connect to WiFi",
        "My Apple ID password is not working",
        "My iPhone screen is not responding to touch",
        "Apple Music won't play my songs",
    ]

    for query in queries:
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
                f"\n--- Result {i} ---"
            )

            print(
                f"Similarity: "
                f"{result['similarity']:.3f}"
            )

            print(
                f"Response quality: "
                f"{result['quality']}"
            )

            print(
                f"Combined score: "
                f"{result['combined_score']:.3f}"
            )

            print("\nHistorical customer:")
            print(result["customer_text"])

            print("\nHistorical AppleSupport response:")
            print(result["response_text"])


if __name__ == "__main__":
    main()