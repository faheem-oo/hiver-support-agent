import re


HIGH_RISK_PATTERNS = {
    "security_or_account": [
        r"\bhacked\b",
        r"\bhack\b",
        r"\bstolen\b",
        r"\bphishing\b",
        r"\bscam\b",
        r"\baccount compromised\b",
        r"\bcan't access my account\b",
        r"\bcannot access my account\b",
    ],

    "data_loss": [
        r"\blost all my data\b",
        r"\blost all my photos\b",
        r"\bphotos are gone\b",
        r"\bmessages are gone\b",
        r"\bdata disappeared\b",
        r"\bdeleted everything\b",
    ],

    "hardware_or_repair": [
        r"\bbroken\b",
        r"\bcracked\b",
        r"\bwon't turn on\b",
        r"\bwill not turn on\b",
        r"\bwon't charge\b",
        r"\bwill not charge\b",
        r"\boverheating\b",
        r"\bvery hot\b",
        r"\brepair\b",
        r"\breplacement\b",
    ],

    "repeated_troubleshooting": [
        r"\btried everything\b",
        r"\btried all\b",
        r"\btried restarting\b",
        r"\brestarted several times\b",
        r"\bstill doesn't work\b",
        r"\bstill does not work\b",
        r"\bstill not working\b",
        r"\bstill isn't working\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b",
    ],

    "severe_failure": [
        r"\bcompletely unusable\b",
        r"\bcan't use my phone\b",
        r"\bcannot use my phone\b",
        r"\bcan't use my iphone\b",
        r"\bcannot use my iphone\b",
        r"\bphone is dead\b",
        r"\biphone is dead\b",
    ],
}


def detect_escalation(message, retrieval_score=None):
    """
    Conservative escalation policy.

    Returns:
        (should_escalate, reason)
    """

    text = str(message).lower()

    # Security/account concerns take highest priority.
    for pattern in HIGH_RISK_PATTERNS["security_or_account"]:
        if re.search(pattern, text):
            return True, "security_or_account_issue"

    # Potential data loss.
    for pattern in HIGH_RISK_PATTERNS["data_loss"]:
        if re.search(pattern, text):
            return True, "possible_data_loss"

    # Hardware / repair cases.
    for pattern in HIGH_RISK_PATTERNS["hardware_or_repair"]:
        if re.search(pattern, text):
            return True, "hardware_or_repair_issue"

    # Severe failures.
    for pattern in HIGH_RISK_PATTERNS["severe_failure"]:
        if re.search(pattern, text):
            return True, "severe_device_failure"

    # Repeated failed troubleshooting.
    for pattern in HIGH_RISK_PATTERNS["repeated_troubleshooting"]:
        if re.search(pattern, text):
            return True, "troubleshooting_already_attempted"

    # Low retrieval confidence means the system lacks evidence
    # for a grounded response.
    if retrieval_score is not None:
        if retrieval_score < 0.25:
            return True, "insufficient_historical_evidence"

    return False, ""


if __name__ == "__main__":
    test_messages = [
        "My iPhone battery is draining very quickly",
        "I tried everything and it still doesn't work",
        "My account has been hacked",
        "My iPhone screen is cracked",
        "My phone is completely unusable",
        "How do I change my wallpaper?",
    ]

    for message in test_messages:
        escalate, reason = detect_escalation(message)

        print("\nMessage:")
        print(message)

        print("Escalate:", escalate)
        print("Reason:", reason)