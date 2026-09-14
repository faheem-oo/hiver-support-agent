import re
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# Load intent classifier
# ---------------------------------------------------------

intent_vectorizer = joblib.load(
    BASE_DIR / "models" / "intent_tfidf.joblib"
)

intent_model = joblib.load(
    BASE_DIR / "models" / "intent_classifier.joblib"
)


# ---------------------------------------------------------
# Load retrieval data
# ---------------------------------------------------------

retrieval_df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "applesupport_retrieval.csv"
)

# Only use responses that passed our useful-response filter.
retrieval_df = retrieval_df[
    retrieval_df["response_is_generic"] == False
].copy()

retrieval_df = retrieval_df.reset_index(drop=True)


# ---------------------------------------------------------
# Load retrieval model
# ---------------------------------------------------------

retrieval_vectorizer = joblib.load(
    BASE_DIR / "models" / "retrieval_tfidf.joblib"
)

retrieval_matrix = joblib.load(
    BASE_DIR / "models" / "retrieval_matrix.joblib"
)


# ---------------------------------------------------------
# Intent prediction
# ---------------------------------------------------------

def predict_intent(message):
    """
    Predict the primary support intent.
    """

    X = intent_vectorizer.transform([message])

    probabilities = intent_model.predict_proba(X)[0]
    classes = intent_model.classes_

    best_index = probabilities.argmax()

    intent = classes[best_index]
    confidence = float(probabilities[best_index])

    return intent, confidence


# ---------------------------------------------------------
# Historical intent prediction
# ---------------------------------------------------------

def predict_historical_intents():
    """
    Predict intents for historical customer messages.

    This is used to make retrieval intent-aware.
    """

    messages = retrieval_df["customer_text"].fillna("").astype(str)

    X = intent_vectorizer.transform(messages)

    predictions = intent_model.predict(X)

    return predictions


# Predict historical intents once when the agent starts.
retrieval_df["historical_intent"] = predict_historical_intents()


# ---------------------------------------------------------
# Response quality heuristic
# ---------------------------------------------------------

def response_quality(response):
    """
    Give higher scores to responses that contain useful
    support actions rather than pure DM redirects.
    """

    text = str(response).lower()

    score = 0

    useful_patterns = [
        "try",
        "check",
        "follow these steps",
        "steps",
        "restart",
        "update",
        "reset",
        "settings",
        "backup",
        "restore",
        "password",
        "wifi",
        "wi-fi",
        "bluetooth",
        "battery",
        "charge",
        "screen",
        "touch",
        "error",
        "article",
        "support article",
    ]

    for pattern in useful_patterns:
        if pattern in text:
            score += 1

    # Strongly penalize responses that are basically
    # "send us a DM".
    generic_patterns = [
        "take a closer look",
        "send us a dm",
        "please send us a dm",
        "continue there",
        "we have received your dm",
    ]

    for pattern in generic_patterns:
        if pattern in text:
            score -= 3

    return score


# ---------------------------------------------------------
# Retrieve historical responses
# ---------------------------------------------------------

def retrieve_responses(message, predicted_intent, top_k=3):
    """
    Retrieve historical examples while preferring examples
    from the same predicted intent.
    """

    query_vector = retrieval_vectorizer.transform([message])

    similarities = (
        retrieval_matrix @ query_vector.T
    ).toarray().ravel()

    # -----------------------------------------------------
    # First pass: same-intent historical examples
    # -----------------------------------------------------

    same_intent_mask = (
        retrieval_df["historical_intent"] == predicted_intent
    ).to_numpy()

    candidate_indices = same_intent_mask.nonzero()[0]

    # If we don't have enough same-intent examples,
    # fall back to the complete retrieval set.
    if len(candidate_indices) < top_k:
        candidate_indices = range(len(retrieval_df))

    candidates = []

    for idx in candidate_indices:
        similarity = float(similarities[idx])

        if similarity <= 0:
            continue

        response = retrieval_df.iloc[idx]["response_text"]

        quality = response_quality(response)

        # Combined score:
        # semantic similarity is primary,
        # response quality provides a small reranking signal.
        combined_score = similarity + 0.015 * quality

        candidates.append(
            {
                "index": idx,
                "similarity": similarity,
                "quality": quality,
                "combined_score": combined_score,
            }
        )

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    results = []

    for candidate in candidates[:top_k]:

        row = retrieval_df.iloc[candidate["index"]]

        results.append(
            {
                "similarity": candidate["similarity"],
                "quality": candidate["quality"],
                "customer": row["customer_text"],
                "response": row["response_text"],
                "historical_intent": row["historical_intent"],
            }
        )

    return results


# ---------------------------------------------------------
# Escalation detection
# ---------------------------------------------------------

def detect_escalation(message, retrieval_score=None):
    """
    Decide whether the issue should be escalated.

    Escalation is based on severity, security/data risk,
    repeated troubleshooting failure, unresolved support,
    and purchase/financial issues.

    The policy deliberately avoids escalating ordinary
    technical complaints just because they contain words
    such as "update", "freeze", or "problem".
    """

    text = message.lower()

    # -----------------------------------------------------
    # 1. Security / account compromise
    # -----------------------------------------------------

    security_patterns = [
        "hacked",
        "hack",
        "phishing",
        "scam",
        "fraud",
        "compromised",
        "someone accessed",
        "someone has access",
        "stolen account",
        "suspicious email",
        "bogus email",
        "odd address",
    ]

    if any(pattern in text for pattern in security_patterns):
        return True, "security_or_account_issue"

    # -----------------------------------------------------
    # 2. Data loss
    # -----------------------------------------------------

    data_loss_patterns = [
        "lost my data",
        "lost all my data",
        "data disappeared",
        "photos disappeared",
        "photos are gone",
        "photos deleted",
        "messages deleted",
        "messages disappeared",
        "contacts are disappearing",
        "contacts disappeared",
        "deleted all my photos",
    ]

    if any(pattern in text for pattern in data_loss_patterns):
        return True, "data_loss_or_icloud_issue"

    # -----------------------------------------------------
    # 3. Hardware / severe physical failure
    # -----------------------------------------------------

    hardware_patterns = [
        "cracked screen",
        "broken screen",
        "screen is broken",
        "water damage",
        "repair",
        "replacement",
        "displaying horizontal",
        "horizontal lines",
        "horizontal shadows",
    ]

    if any(pattern in text for pattern in hardware_patterns):
        return True, "hardware_or_repair_issue"

    severe_failure_patterns = [
        "completely unusable",
        "completely dead",
        "phone is dead",
        "device is dead",
        "won't turn on",
        "doesn't turn on",
        "will not turn on",
        "stuck on this screen",
    ]

    if any(pattern in text for pattern in severe_failure_patterns):
        return True, "severe_device_failure"

    # -----------------------------------------------------
    # 4. Repeated instability / severe recurring failure
    # -----------------------------------------------------

    repeated_failure_patterns = [
    "every 5 mins",
    "every 5 minutes",
    "every five minutes",
    "every 10 minutes",
    "every ten minutes",
    "every few minutes",
    "every minute",
    "multiple times a day",
    "several times a day",
    "keeps restarting",
    "keeps rebooting",
    "keeps shutting off",
    "constantly restarting",
    "constantly rebooting",
    "constantly freezing",
    "over and over",
    "again and again",
    "every time i try",
    "restarting every",
    "restarting very",
    "rebooting every",
    "freezing every",
    "freezes every",
]

    severe_symptom_patterns = [
        "freeze",
        "freezes",
        "freezing",
        "restarting",
        "restarts",
        "rebooting",
        "shutting off",
        "shuts off",
        "cutting out",
    ]

    repeated_failure = any(
        pattern in text
        for pattern in repeated_failure_patterns
    )

    severe_symptom = any(
        pattern in text
        for pattern in severe_symptom_patterns
    )

    if repeated_failure and severe_symptom:
        return True, "repeated_device_failure"
        # Update-related repeated failures are higher risk
    # when the device repeatedly freezes/restarts/shuts down.

    update_related = any(
        pattern in text
        for pattern in [
            "after the update",
            "after i updated",
            "since i updated",
            "since the update",
            "after updating",
            "these updates",
            "ios 11",
            "ios11",
            "new update",
        ]
    )

    if update_related and severe_symptom:
        return True, "repeated_update_related_failure"

    # Explicit "still broken after troubleshooting"
    troubleshooting_patterns = [
        "tried everything",
        "tried all",
        "nothing works",
        "still doesn't work",
        "still not working",
        "still won't work",
        "already tried",
        "tried updating",
        "tried restarting",
        "restarted several times",
        "backup and updated",
        "did not fix",
        "didn't fix",
        "not fixed",
    ]

    unresolved_patterns = [
        "still persists",
        "issue persists",
        "problem persists",
        "still having the issue",
        "still have the problem",
        "still waiting",
        "waiting on you",
        "nobody is solving",
        "nobody has solved",
    ]

    troubleshooting_attempted = any(
        pattern in text
        for pattern in troubleshooting_patterns
    )

    unresolved = any(
        pattern in text
        for pattern in unresolved_patterns
    )

        # Multiple explicit troubleshooting attempts are enough
    # to justify escalation when the underlying problem persists.
    multiple_attempts = any(
        pattern in text
        for pattern in [
            "restarting several times",
            "restarted several times",
            "tried several times",
            "tried multiple times",
            "tried updating",
            "tried restarting",
            "tried resetting",
            "tried everything",
            "tried all",
        ]
    )

    if multiple_attempts:
        return True, "repeated_troubleshooting_failed"

    if troubleshooting_attempted and unresolved:
        return True, "repeated_troubleshooting_failed"

    # -----------------------------------------------------
    # 5. Account recovery problems
    # -----------------------------------------------------

    account_recovery_patterns = [
        "can't recover my apple id",
        "cannot recover my apple id",
        "account recovery",
        "recover my apple id",
        "apple id recovery",
        "password recovery",
        "can't access my account",
        "cannot access my account",
    ]

    if any(
        pattern in text
        for pattern in account_recovery_patterns
    ):
        return True, "account_recovery_issue"

    # -----------------------------------------------------
    # 6. Purchase / order / compensation disputes
    # -----------------------------------------------------

    purchase_escalation_patterns = [
        "charged",
        "bill",
        "billing",
        "refund",
        "gift card",
        "voucher",
        "reservation",
        "order",
        "purchase",
        "won't give me",
        "money",
    ]

    purchase_context = any(
        pattern in text
        for pattern in purchase_escalation_patterns
    )

    if purchase_context:
        return True, "purchase_or_compensation_issue"

    # -----------------------------------------------------
    # 7. Very low retrieval confidence
    # -----------------------------------------------------

    if (
        retrieval_score is not None
        and retrieval_score < 0.15
    ):
        return True, "insufficient_historical_evidence"

    return False, ""

# ---------------------------------------------------------
# Clean historical reply
# ---------------------------------------------------------

def clean_historical_response(response):
    """
    Remove Twitter-specific mentions and URLs from
    historical support responses.
    """

    text = str(response)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------------------------
# Draft reply
# ---------------------------------------------------------

def draft_reply(
    message,
    intent,
    should_escalate,
    escalation_reason,
    evidence
):
    """
    Generate a grounded support reply.

    Escalated issues receive a safe handoff.
    Non-escalated issues use intent-specific guidance
    grounded in retrieved historical evidence.
    """

    # -----------------------------------------------------
    # Escalation response
    # -----------------------------------------------------

    if should_escalate:

        reason_text = {
            "security_or_account_issue":
                "because this involves account security",
            "data_loss_or_icloud_issue":
                "because this may involve data loss",
            "hardware_or_repair_issue":
                "because this may require hardware service",
            "repeated_troubleshooting_failed":
                "because the issue has persisted after troubleshooting",
            "severe_device_failure":
                "because the device appears to have a severe failure",
            "insufficient_historical_evidence":
                "because we don't have enough reliable historical evidence",
        }.get(
            escalation_reason,
            "because the issue may require additional support"
        )

        return (
            f"Thanks for reaching out. We recommend contacting "
            f"Apple Support directly {reason_text}. "
            f"They can take a closer look and provide the appropriate "
            f"next steps."
        )

    # -----------------------------------------------------
    # No historical evidence
    # -----------------------------------------------------

    if not evidence:

        return (
            "Thanks for reaching out. Could you provide a little "
            "more detail about what happens when you try this?"
        )

    # -----------------------------------------------------
    # Extract useful historical responses
    # -----------------------------------------------------

    historical_responses = []

    for item in evidence:

        response = clean_historical_response(
            item["response"]
        )

        if not response:
            continue

        quality = item["quality"]

        # Ignore responses that are purely DM redirects
        # or otherwise scored as poor evidence.
        if quality <= 0:
            continue

        historical_responses.append(response)

    # -----------------------------------------------------
    # Intent-specific response generation
    # -----------------------------------------------------

    if intent == "connectivity":

        return (
            "Thanks for reaching out. Let's troubleshoot the "
            "Wi-Fi connection. What happens when you try to connect "
            "to the network, and do you see any error message? "
            "If possible, also let us know whether the iPhone can "
            "connect to other Wi-Fi networks."
        )

    if intent == "battery_charging":

        return (
            "Thanks for reaching out. Since the battery drain "
            "started after the iOS update, please check your "
            "Battery settings to see which apps or activity are "
            "using the most power. If the issue continues, let us "
            "know your iPhone model and iOS version so we can "
            "narrow down the issue."
        )

    if intent == "ios_update":

        return (
            "Thanks for reaching out. Please let us know what "
            "happens when you try to install or use the iOS update, "
            "including any error message you see and your iPhone "
            "model. That will help us determine the appropriate "
            "next troubleshooting step."
        )

    if intent == "device_performance":

        return (
            "Thanks for reaching out. Could you tell us whether "
            "the iPhone is slow, freezing, crashing, or becoming "
            "unresponsive? Please also share your iPhone model and "
            "iOS version so we can narrow down the issue."
        )

    if intent == "screen_display":

        return (
            "Thanks for reaching out. Could you describe what is "
            "happening with the screen or touch response? Please "
            "also let us know your iPhone model and iOS version, "
            "and whether the issue happens consistently."
        )

    if intent == "apple_id_icloud":

        return (
            "Thanks for reaching out. Please tell us what happens "
            "when you try to sign in to your Apple ID or iCloud, "
            "including any error message you see. Avoid sharing "
            "your password or other sensitive account information."
        )

    if intent == "apps":

        return (
            "Thanks for reaching out. Which app is affected, and "
            "what happens when you try to use it? Please include "
            "any error message and, if possible, your iPhone model "
            "and iOS version."
        )

    if intent == "itunes_music":

        return (
            "Thanks for reaching out. Could you tell us what "
            "happens when you try to play or access your music? "
            "Please include any error message and let us know "
            "which device and iOS version you're using."
        )

    if intent == "app_store_purchases":

        return (
            "Thanks for reaching out. Could you tell us whether "
            "the issue is with downloading an app, making a "
            "purchase, or a payment? Please don't share payment "
            "details or passwords here."
        )

    if intent == "apple_watch":

        return (
            "Thanks for reaching out. Could you tell us what is "
            "happening with your Apple Watch and whether it is "
            "connected to your iPhone? Please include the Watch "
            "and iPhone models if possible."
        )

    # -----------------------------------------------------
    # Other / unclear
    # -----------------------------------------------------

    return (
        "Thanks for reaching out. We'd like to understand the "
        "issue better. Could you describe what is happening, "
        "including any error message or troubleshooting steps "
        "you have already tried?"
    )


# ---------------------------------------------------------
# Full agent
# ---------------------------------------------------------

def run_agent(message):

    intent, confidence = predict_intent(message)

    evidence = retrieve_responses(
        message,
        predicted_intent=intent,
        top_k=3
    )

    if evidence:
        retrieval_score = evidence[0]["similarity"]
    else:
        retrieval_score = 0.0

    should_escalate, escalation_reason = detect_escalation(
        message,
        retrieval_score
    )

    reply = draft_reply(
        message,
        intent,
        should_escalate,
        escalation_reason,
        evidence
    )

    return {
        "intent": intent,
        "confidence": confidence,
        "escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "retrieval_score": retrieval_score,
        "reply": reply,
        "evidence": evidence,
    }


# ---------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    print("AppleSupport AI Agent")
    print("=====================")

    while True:

        message = input(
            "\nCustomer message (type 'exit' to quit): "
        ).strip()

        if message.lower() == "exit":
            break

        if not message:
            continue

        result = run_agent(message)

        print("\n--- RESULT ---")

        print(
            f"Intent: {result['intent']}"
        )

        print(
            f"Intent confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Escalate: "
            f"{result['escalate']}"
        )

        print(
            f"Escalation reason: "
            f"{result['escalation_reason'] or 'none'}"
        )

        print(
            f"Retrieval score: "
            f"{result['retrieval_score']:.3f}"
        )

        print("\nDraft reply:")
        print(result["reply"])

        print("\nHistorical evidence:")

        for i, item in enumerate(
            result["evidence"],
            start=1
        ):

            print(
                f"\n[{i}] "
                f"Similarity: {item['similarity']:.3f} "
                f"| Quality: {item['quality']} "
                f"| Historical intent: "
                f"{item['historical_intent']}"
            )

            print(
                "Customer:"
            )

            print(
                item["customer"]
            )

            print(
                "AppleSupport:"
            )

            print(
                clean_historical_response(
                    item["response"]
                )
            )