import re
import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/applesupport_threads.csv"
OUTPUT_FILE = "data/processed/applesupport_retrieval.csv"


GENERIC_PATTERNS = [
    r"\bplease dm us\b",
    r"\bsend us a dm\b",
    r"\bjoin us in dm\b",
    r"\bwe're here to help\b",
    r"\bwe are here to help\b",
    r"\bwe'd love to help\b",
    r"\bwe would love to help\b",
    r"\bwe're happy to help\b",
    r"\bwe are happy to help\b",
    r"\bwe'll be happy to help\b",
    r"\bwe will be happy to help\b",
    r"\bthank you for reaching out\b",
    r"\bthanks for reaching out\b",
]


def is_generic_response(text):
    text = str(text).lower()

    # Remove URLs before checking response content.
    text = re.sub(r"https?://\S+", "", text)

    # If the response contains a useful instruction/question,
    # don't automatically classify it as generic.
    useful_patterns = [
        r"\btry\b",
        r"\bcheck\b",
        r"\bsettings\b",
        r"\bupdate\b",
        r"\brestart\b",
        r"\breset\b",
        r"\bverify\b",
        r"\bsteps\b",
        r"\barticle\b",
        r"\btap\b",
        r"\bgo to\b",
        r"\bturn\b",
        r"\binstall\b",
        r"\bremove\b",
        r"\bbackup\b",
        r"\brestore\b",
        r"\bdiagnos\b",
    ]

    if any(re.search(p, text) for p in useful_patterns):
        return False

    matches = sum(
        bool(re.search(p, text))
        for p in GENERIC_PATTERNS
    )

    return matches > 0


def main():
    print("Loading AppleSupport threads...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total turns: {len(df):,}")

    # Sort to guarantee chronological turn order.
    df = df.sort_values(
        ["conversation_id", "turn_number"]
    ).reset_index(drop=True)

    examples = []

    # Build customer -> next AppleSupport response pairs.
    for conversation_id, group in df.groupby(
        "conversation_id",
        sort=False
    ):
        group = group.sort_values("turn_number")

        rows = group.to_dict("records")

        for i in range(len(rows) - 1):
            customer = rows[i]
            response = rows[i + 1]

            # Customer turn followed immediately by AppleSupport.
            if (
                customer["inbound"] is True
                and response["inbound"] is False
            ):
                customer_text = str(customer["text"]).strip()
                response_text = str(response["text"]).strip()

                if not customer_text or not response_text:
                    continue

                examples.append(
                    {
                        "conversation_id": conversation_id,
                        "customer_tweet_id": customer["tweet_id"],
                        "response_tweet_id": response["tweet_id"],
                        "customer_text": customer_text,
                        "response_text": response_text,
                        "response_is_generic": is_generic_response(
                            response_text
                        ),
                    }
                )

    result = pd.DataFrame(examples)

    print(
        f"\nCustomer -> AppleSupport pairs: "
        f"{len(result):,}"
    )

    print("\n=== Response type ===")

    print(
        result["response_is_generic"]
        .value_counts()
        .rename(
            index={
                False: "useful_candidate",
                True: "generic_candidate",
            }
        )
        .to_string()
    )

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n=== Sample useful responses ===")

    useful = result[
        result["response_is_generic"] == False
    ]

    for _, row in useful.head(5).iterrows():
        print("\nCUSTOMER:")
        print(row["customer_text"])

        print("\nAPPLE SUPPORT:")
        print(row["response_text"])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()