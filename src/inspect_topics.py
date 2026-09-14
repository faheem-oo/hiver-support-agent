from pathlib import Path
import pandas as pd
import re

INPUT = Path("data/processed/applesupport_customer_clean.csv")

TOPICS = {
    "ios_update": [
        "ios", "update", "updated", "upgrade", "software"
    ],
    "battery_charging": [
        "battery", "charging", "charge", "drain", "power"
    ],
    "screen_display": [
        "screen", "display", "touch", "brightness"
    ],
    "apps": [
        "app", "apps", "application", "crash", "freezing"
    ],
    "apple_music": [
        "music", "apple music", "itunes", "song", "playlist"
    ],
    "wifi_connectivity": [
        "wifi", "wi-fi", "bluetooth", "connection", "internet"
    ],
    "apple_id_account": [
        "apple id", "icloud", "password", "account",
        "login", "sign in"
    ],
    "device_hardware": [
        "iphone", "ipad", "macbook", "home button",
        "speaker", "camera"
    ],
    "performance": [
        "slow", "freezing", "lag", "stuck", "hang"
    ],
    "purchases_store": [
        "store", "purchase", "buy", "download", "app store"
    ],
    "siri": [
        "siri"
    ],
    "apple_watch": [
        "watch", "apple watch"
    ],
}


def main():
    if not INPUT.exists():
        print(f"File not found: {INPUT}")
        return

    df = pd.read_csv(INPUT)

    print(f"Loaded {len(df):,} customer messages\n")

    text_col = "text"

    for topic, keywords in TOPICS.items():
        pattern = "|".join(
            re.escape(keyword) for keyword in keywords
        )

        mask = df[text_col].fillna("").str.contains(
            pattern,
            case=False,
            regex=True
        )

        matches = df[mask]

        print("=" * 80)
        print(f"{topic.upper()} — {len(matches):,} matches")
        print("=" * 80)

        if len(matches) == 0:
            print("No examples found.\n")
            continue

        sample_size = min(10, len(matches))

        for text in matches[text_col].sample(
            sample_size,
            random_state=42
        ):
            print(f"- {text}")

        print()


if __name__ == "__main__":
    main()