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

    Useful historical responses are ranked above generic
    DM-only responses.
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

                # -------------------------------------------------
        # Topic keyword overlap
        # -------------------------------------------------

        customer_text = str(
            retrieval_df.iloc[idx]["customer_text"]
        ).lower()

        query_text = str(message).lower()

        topic_terms = [
            "wifi",
            "wi-fi",
            "bluetooth",
            "battery",
            "charging",
            "screen",
            "display",
            "touch",
            "ios",
            "update",
            "apple id",
            "icloud",
            "music",
            "itunes",
            "app store",
            "purchase",
            "refund",
            "password",
            "login",
            "iphone",
            "ipad",
            "mac",
            "apple watch",
        ]

        overlapping_terms = [
            term
            for term in topic_terms
            if term in query_text
            and term in customer_text
        ]

        topic_overlap = len(overlapping_terms)

        # -------------------------------------------------
        # Detect generic DM-only responses
        # -------------------------------------------------

        response_lower = str(response).lower()

        generic_response = any(
            pattern in response_lower
            for pattern in [
                "continue working with you via dm",
                "reach out using the link",
                "let's meet in dm",
                "please answer in dm",
                "let us know in a dm",
                "contact us in dm",
                "send us a dm",
                "message us in dm",
            ]
        )

        if generic_response:
            generic_penalty = 0.10
        else:
            generic_penalty = 0.0

        # -------------------------------------------------
        # Combined retrieval score
        # -------------------------------------------------

        combined_score = (
            similarity
            + 0.04 * quality
            + 0.08 * topic_overlap
            - generic_penalty
        )

        candidates.append(
            {
                "index": idx,
                "similarity": similarity,
                "quality": quality,
                "combined_score": combined_score,
            }
        )

    # -----------------------------------------------------
    # Rank candidates
    # -----------------------------------------------------

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    # -----------------------------------------------------
    # Build final results
    # -----------------------------------------------------

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
    escalated,
    escalation_reason,
    evidence,
    retrieval_score=0.0
):
    """
    Generate a support reply using the predicted intent and
    the strongest historical evidence.

    The reply deliberately avoids inventing fixes that are not
    supported by the historical evidence.
    """

    # ---------------------------------------------------------
    # Escalated cases
    # ---------------------------------------------------------

    if escalated:
        reason_messages = {
            "security_or_account_issue":
                "Because this may involve account security, we recommend contacting Apple Support directly so they can verify the account safely.",

            "data_loss_or_icloud_issue":
                "Because this may involve lost data or iCloud information, Apple Support should review the case directly before further troubleshooting.",

            "hardware_or_repair_issue":
                "Because this may involve hardware damage or a repair issue, Apple Support can assess the device and advise on the appropriate service.",

            "severe_device_failure":
                "Because the device appears to have a severe failure, Apple Support should assess it directly.",

            "repeated_device_failure":
                "Because the problem is recurring despite previous attempts, Apple Support should review the case directly.",

            "repeated_update_related_failure":
                "Because the problem appears to be recurring after an update, Apple Support should review the device and update history directly.",

            "repeated_troubleshooting_failed":
                "Because you've already tried several troubleshooting steps without resolving the problem, Apple Support should review the case directly.",

            "account_recovery_issue":
                "Because this involves account recovery, Apple Support can help verify the account and provide the appropriate recovery steps.",

            "purchase_or_compensation_issue":
                "Because this involves a purchase, payment, refund, or compensation concern, Apple Support should review the transaction directly.",

            "multiple_critical_issues":
                "Because there are multiple significant issues involved, Apple Support should review the case directly."
        }

        return reason_messages.get(
            escalation_reason,
            "This issue needs additional support review, so we recommend contacting Apple Support directly."
        )


    # ---------------------------------------------------------
    # Extract useful historical responses
    # ---------------------------------------------------------

    historical_responses = []

    for item in evidence:
        response = item.get("response", "").strip()

        if not response:
            continue

        response_lower = response.lower()

        # Ignore responses that only redirect to DM
        generic_patterns = [
            "continue working with you via dm",
            "reach out using the link",
            "let's meet in dm",
            "please answer in dm",
            "let us know in a dm",
            "contact us in dm"
        ]

        if any(pattern in response_lower for pattern in generic_patterns):
            continue

        historical_responses.append(response)


    # ---------------------------------------------------------
    # If retrieval evidence is weak
    # ---------------------------------------------------------

    if retrieval_score < 0.15 or not historical_responses:
        return (
            "Thanks for reaching out. Could you share a little more "
            "about what is happening, including any error message and "
            "your device model or software version? That will help us "
            "identify the appropriate next step."
        )


    # ---------------------------------------------------------
    # Intent-specific replies
    # ---------------------------------------------------------

    if intent == "connectivity":
        return (
            "Thanks for reaching out. Let's troubleshoot the connection. "
            "Is the problem happening on all networks or only a specific "
            "Wi-Fi/Bluetooth connection? If possible, let us know what "
            "happens when you try to connect and whether you see an error."
        )


    if intent == "battery_charging":
      return (
        "Thanks for reaching out. Let's narrow down the battery or "
        "charging issue. Is the battery draining quickly, charging "
        "slowly, or failing to charge? Please share your iPhone model "
        "and iOS version, and let us know when you first noticed the "
        "problem."
    )


    if intent == "ios_update":
        return (
            "Thanks for reaching out. What happens when you try to install "
            "or use the iOS update? If there is an error message, please "
            "share it along with your iPhone model and current iOS version. "
            "If the problem started after updating, please mention that too."
        )


    if intent == "device_performance":
        return (
            "Thanks for reaching out. Let's narrow down the device issue. "
            "Is the iPhone slow, freezing, restarting, crashing, or becoming "
            "unresponsive? Please share the iPhone model, iOS version, and "
            "whether the problem happens repeatedly."
        )


    if intent == "screen_display":
        return (
            "Thanks for reaching out. Could you describe what is happening "
            "with the display or touch screen? Please let us know whether "
            "the issue is constant or intermittent, and share your device "
            "model and iOS version."
        )


    if intent == "apple_id_icloud":
        return (
            "Thanks for reaching out. Is the issue with signing in, your "
            "Apple ID password, iCloud, or account access? If you see an "
            "error message, please share the message. For your security, "
            "please don't send your password or other sensitive account "
            "information."
        )


    if intent == "apps":
        return (
            "Thanks for reaching out. Which app is affected, and what "
            "happens when you use it? Please share any error message, "
            "your device model, and iOS version so we can narrow down "
            "the issue."
        )


    if intent == "itunes_music":
        return (
            "Thanks for reaching out. Is the problem with missing songs, "
            "music playback, downloads, or your music library? Please "
            "describe what happens and share your device model and iOS "
            "version."
        )


    if intent == "app_store_purchases":
        return (
            "Thanks for reaching out. Is the issue with downloading an "
            "app, making a purchase, a payment, or a refund? Please share "
            "the error or message you see. For your security, don't send "
            "your password or full payment details."
        )


    if intent == "apple_watch":
        return (
            "Thanks for reaching out. What problem are you experiencing "
            "with your Apple Watch? Please let us know whether it involves "
            "the Watch itself, pairing, or connectivity, and share the "
            "Watch/iPhone models if possible."
        )


    # ---------------------------------------------------------
    # Other / unclear
    # ---------------------------------------------------------

    return (
        "Thanks for reaching out. Could you describe the problem in a "
        "little more detail, including what you expected to happen, what "
        "actually happens, and any error message you see? Your device "
        "model and software version would also help us narrow it down."
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
    evidence,
    retrieval_score
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