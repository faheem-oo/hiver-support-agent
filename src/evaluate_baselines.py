import pandas as pd
import re

from sklearn.metrics import accuracy_score, f1_score, classification_report


PATH = "data/processed/golden_set.csv"

df = pd.read_csv(PATH, keep_default_na=False)


# ============================================================
# BASELINE 1: MAJORITY CLASS
# ============================================================

majority_intent = df["intent"].value_counts().idxmax()

df["majority_prediction"] = majority_intent


print("=" * 60)
print("BASELINE 1: MAJORITY CLASS")
print("=" * 60)

print("Majority intent:", majority_intent)

print(
    "Accuracy:",
    round(
        accuracy_score(df["intent"], df["majority_prediction"]),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            df["intent"],
            df["majority_prediction"],
            average="macro",
            zero_division=0
        ),
        4
    )
)

print("\nClassification report:")
print(
    classification_report(
        df["intent"],
        df["majority_prediction"],
        zero_division=0
    )
)


# ============================================================
# BASELINE 2: KEYWORD RULES
# ============================================================

def keyword_classifier(text):

    text = text.lower()

    # Apple Watch
    if re.search(r"\bapple watch\b|\bwatchos\b", text):
        return "apple_watch"

    # Purchases / billing / App Store
    if re.search(
        r"\bapp store\b|purchase|purchased|refund|refunds|"
        r"charged|charge|billing|payment|gift card|order|"
        r"reservation|voucher",
        text
    ):
        return "app_store_purchases"

    # Apple Music / iTunes
    if re.search(
        r"\bapple music\b|\bitunes\b|\bmusic\b|\bsong\b|"
        r"\bsongs\b|\bplaylist\b|\bpodcast\b",
        text
    ):
        return "itunes_music"

    # Battery / charging
    if re.search(
        r"\bbattery\b|charging|charger|charge|drain|"
        r"low power|dies|battery life",
        text
    ):
        return "battery_charging"

    # Screen / display
    if re.search(
        r"\bscreen\b|\bdisplay\b|touch|brightness|"
        r"black screen|flicker|lines|shadow",
        text
    ):
        return "screen_display"

    # Connectivity
    if re.search(
        r"\bwi[- ]?fi\b|bluetooth|cellular|network|"
        r"internet|connection|connect|no service|"
        r"hotspot",
        text
    ):
        return "connectivity"

    # Apple ID / iCloud
    if re.search(
        r"apple id|icloud|password|sign in|sign-in|"
        r"login|account|activation|2fa|two factor|"
        r"lost mode",
        text
    ):
        return "apple_id_icloud"

    # iOS updates
    if re.search(
        r"\bios\b.*\bupdate\b|\bios\s*\d|"
        r"\bupdate\b|\bupdated\b|\bupdating\b|"
        r"\bupgrade\b|\bdowngrade\b|software update|"
        r"high sierra",
        text
    ):
        return "ios_update"

    # Device performance
    if re.search(
        r"\bslow\b|lag|lagging|freeze|freezes|"
        r"freezing|crash|crashes|stuck|unresponsive|"
        r"restart|restarts|reboot|reboots|"
        r"shuts off|shut off|reset",
        text
    ):
        return "device_performance"

    # Apps
    if re.search(
        r"\bapp\b|\bapps\b|application|"
        r"twitter app|maps app|messaging app",
        text
    ):
        return "apps"

    return "other"


df["keyword_prediction"] = df["clean_text"].apply(keyword_classifier)


print("\n")
print("=" * 60)
print("BASELINE 2: KEYWORD RULES")
print("=" * 60)

print(
    "Accuracy:",
    round(
        accuracy_score(
            df["intent"],
            df["keyword_prediction"]
        ),
        4
    )
)

print(
    "Macro F1:",
    round(
        f1_score(
            df["intent"],
            df["keyword_prediction"],
            average="macro",
            zero_division=0
        ),
        4
    )
)

print("\nClassification report:")
print(
    classification_report(
        df["intent"],
        df["keyword_prediction"],
        zero_division=0
    )
)


# ============================================================
# SHOW MISCLASSIFIED EXAMPLES
# ============================================================

errors = df[
    df["intent"] != df["keyword_prediction"]
][
    [
        "id",
        "clean_text",
        "intent",
        "keyword_prediction"
    ]
]

print("\n")
print("=" * 60)
print("KEYWORD BASELINE — SAMPLE ERRORS")
print("=" * 60)

print(
    errors.head(20).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "majority_intent": majority_intent,
    "majority_accuracy": accuracy_score(
        df["intent"],
        df["majority_prediction"]
    ),
    "majority_macro_f1": f1_score(
        df["intent"],
        df["majority_prediction"],
        average="macro",
        zero_division=0
    ),
    "keyword_accuracy": accuracy_score(
        df["intent"],
        df["keyword_prediction"]
    ),
    "keyword_macro_f1": f1_score(
        df["intent"],
        df["keyword_prediction"],
        average="macro",
        zero_division=0
    ),
}

pd.DataFrame([results]).to_csv(
    "outputs/baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("outputs/baseline_results.csv")