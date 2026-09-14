import re
import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/applesupport_training_messages.csv"
GOLDEN_FILE = "data/processed/golden_set.csv"
OUTPUT_FILE = "data/processed/weak_labels.csv"


RULES = {
    "apple_watch": [
        r"\bapple watch\b",
        r"\bwatchos\b",
    ],

    "app_store_purchases": [
        r"\bapp store\b",
        r"\bitunes store\b",
        r"\bgift card\b",
        r"\bpurchase\b",
        r"\bpurchased\b",
        r"\brefund\b",
        r"\bcharged\b",
        r"\bbilling\b",
        r"\bpayment\b",
        r"\border\b",
    ],

    "itunes_music": [
        r"\bapple music\b",
        r"\bitunes\b",
        r"\bmusic\b",
        r"\bsong\b",
        r"\bplaylist\b",
        r"\bpodcast\b",
    ],

    "battery_charging": [
        r"\bbattery\b",
        r"\bcharge\b",
        r"\bcharging\b",
        r"\bcharger\b",
        r"\blow power\b",
        r"\bbattery drain\b",
        r"\bdrain\b",
        r"\bdies\b",
    ],

    "screen_display": [
        r"\bscreen\b",
        r"\bdisplay\b",
        r"\btouch screen\b",
        r"\btouchscreen\b",
        r"\bbrightness\b",
        r"\bblack screen\b",
        r"\bflicker\b",
        r"\bflickering\b",
    ],

    "connectivity": [
        r"\bwifi\b",
        r"\bwi-fi\b",
        r"\bbluetooth\b",
        r"\bcellular\b",
        r"\bnetwork\b",
        r"\binternet\b",
        r"\bno service\b",
        r"\bconnection\b",
        r"\bconnect\b",
    ],

    "apple_id_icloud": [
        r"\bapple id\b",
        r"\bicloud\b",
        r"\bpassword\b",
        r"\bsign in\b",
        r"\bsign-in\b",
        r"\blogin\b",
        r"\blog in\b",
        r"\baccount\b",
        r"\bactivation\b",
        r"\blost mode\b",
        r"\b2fa\b",
        r"\btwo-factor\b",
    ],

    "ios_update": [
        r"\bios\s*\d+",
        r"\bios update\b",
        r"\bsoftware update\b",
        r"\bupdate\b",
        r"\bupdated\b",
        r"\bupdating\b",
        r"\bupgrade\b",
        r"\bupgraded\b",
        r"\bdowngrade\b",
        r"\bhigh sierra\b",
    ],

    "apps": [
        r"\bapp\b",
        r"\bapps\b",
        r"\bapplication\b",
        r"\bapplications\b",
    ],

    "device_performance": [
        r"\bslow\b",
        r"\blag\b",
        r"\blagging\b",
        r"\bfreeze\b",
        r"\bfreezes\b",
        r"\bfreezing\b",
        r"\bfrozen\b",
        r"\bcrash\b",
        r"\bcrashes\b",
        r"\bcrashing\b",
        r"\bstuck\b",
        r"\bunresponsive\b",
        r"\brestart\b",
        r"\brestarting\b",
        r"\breboot\b",
        r"\brebooting\b",
        r"\bshut off\b",
        r"\bshuts off\b",
        r"\breset\b",
    ],
}


def matches(text, patterns):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify(text):
    """
    High-precision weak labeling.

    We deliberately discard ambiguous messages instead of
    assigning a potentially incorrect pseudo-label.
    """

    text = str(text).lower()

    matched = []

    for intent, patterns in RULES.items():
        if matches(text, patterns):
            matched.append(intent)

    # No rule matched
    if not matched:
        return None, "unmatched"

    # Special handling:
    # If a message explicitly says an update caused a symptom,
    # prefer the symptom as the primary intent.
    symptom_intents = {
        "battery_charging",
        "screen_display",
        "connectivity",
        "device_performance",
        "apps",
    }

    if "ios_update" in matched:
        symptom_matches = [
            x for x in matched
            if x in symptom_intents
        ]

        if len(symptom_matches) == 1:
            return symptom_matches[0], "update_caused_symptom"

    # If exactly one intent matched, high confidence
    if len(matched) == 1:
        return matched[0], "single_rule"

    # Multiple unrelated rules matched -> discard
    return None, "ambiguous"


def main():
    print("Loading training messages...")

    df = pd.read_csv(INPUT_FILE)

    golden = pd.read_csv(GOLDEN_FILE)

    print(f"Training messages: {len(df):,}")
    print(f"Golden messages: {len(golden):,}")

    # Exclude golden examples from weak-label training.
    golden_tweet_ids = set(
        golden["tweet_id"].astype(str)
    )

    golden_clean_texts = set(
        golden["clean_text"]
        .fillna("")
        .astype(str)
    )

    df["tweet_id"] = df["tweet_id"].astype(str)

    before = len(df)

    df = df[
        ~df["tweet_id"].isin(golden_tweet_ids)
    ].copy()

    df = df[
        ~df["clean_text"].fillna("").astype(str).isin(
            golden_clean_texts
        )
    ].copy()

    print(
        f"Removed {before - len(df):,} "
        "golden-set/duplicate examples"
    )

    print("\nCreating weak labels...")

    labels = df["clean_text"].apply(classify)

    df["weak_intent"] = labels.apply(lambda x: x[0])
    df["label_rule"] = labels.apply(lambda x: x[1])

        # Keep high-confidence intent labels.
    weak = df[
        df["weak_intent"].notna()
    ].copy()

    # Messages with no recognized intent become weak "other" examples.
    # We intentionally do NOT include ambiguous multi-intent messages.
    unmatched = df[
        (df["weak_intent"].isna())
        & (df["label_rule"] == "unmatched")
    ].copy()

    unmatched["weak_intent"] = "other"

    weak = pd.concat(
        [weak, unmatched],
        ignore_index=True
    )

    # Save useful columns
    weak = weak[
        [
            "tweet_id",
            "text",
            "clean_text",
            "weak_intent",
            "label_rule",
        ]
    ]

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    weak.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nWeak-labeled examples: {len(weak):,}")

    print("\n=== Weak-label distribution ===")

    counts = (
        weak["weak_intent"]
        .value_counts()
        .sort_index()
    )

    print(counts.to_string())

    print("\n=== Label rule distribution ===")

    print(
        weak["label_rule"]
        .value_counts()
        .to_string()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()