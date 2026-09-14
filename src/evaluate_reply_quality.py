import os
import json
import argparse
import pandas as pd
from pathlib import Path


RUBRIC = {
    "relevance": "Does the reply directly address the customer's actual problem?",
    "groundedness": "Does the reply stay faithful to information in the customer's message and avoid invented assumptions?",
    "helpfulness": "Does the reply provide a useful next step, diagnostic question, or appropriate handoff?",
    "specificity": "Does the reply use important details already present in the customer's message instead of giving a generic response?",
    "safety_escalation": "Is the response appropriately cautious and does it escalate when the situation reasonably requires human support?"
}


def build_prompt(row):
    return f"""
You are evaluating an AI customer-support agent for AppleSupport.

Customer message:
{row['text']}

Predicted intent:
{row['predicted_intent']}

Predicted escalation:
{row['predicted_escalate']}

Escalation reason:
{row.get('escalation_reason', '')}

Draft reply:
{row['draft_reply']}

Evaluate ONLY the draft reply against the customer's message.

Score each dimension from 1 to 5:

1. relevance:
1 = unrelated
3 = partially addresses the issue
5 = directly addresses the actual issue

2. groundedness:
1 = contains major unsupported assumptions or incorrect claims
3 = mostly grounded but has some assumptions
5 = fully grounded in the customer's message

3. helpfulness:
1 = provides no useful next step
3 = somewhat useful
5 = gives a clear and appropriate next step or handoff

4. specificity:
1 = generic and ignores important context
3 = uses some context
5 = meaningfully uses the details supplied by the customer

5. safety_escalation:
1 = inappropriate or unsafe handling
3 = debatable handling
5 = appropriately cautious and escalates when warranted

Also provide:
- overall_score: 1-5
- pass: true if overall_score >= 4, otherwise false
- short_reason: one concise explanation

Return ONLY valid JSON in this format:

{{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "specificity": 1,
  "safety_escalation": 1,
  "overall_score": 1,
  "pass": false,
  "short_reason": "..."
}}
"""


def evaluate_with_openai(rows, model):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI package is not installed. Run: pip install openai"
        )

    client = OpenAI()

    results = []

    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        print(f"Evaluating {i}/{len(rows)}...", flush=True)

        response = client.responses.create(
            model=model,
            input=build_prompt(row)
        )

        text = response.output_text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {
                "parse_error": True,
                "raw_response": text
            }

        result["id"] = int(row["id"])
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/heldout_results.csv"
    )
    parser.add_argument(
        "--output",
        default="outputs/reply_quality_results.csv"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.6-mini"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required = [
        "id",
        "text",
        "predicted_intent",
        "predicted_escalate",
        "draft_reply"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} held-out examples.")

    if not os.getenv("OPENAI_API_KEY"):
        print()
        print("OPENAI_API_KEY is not set.")
        print("No LLM-judge scores were generated.")
        print("This is intentional: the harness will not fabricate results.")
        print()
        print("To run the judge:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  python src/evaluate_reply_quality.py")
        return

    results = evaluate_with_openai(df, args.model)

    result_df = pd.DataFrame(results)

    merged = df.merge(result_df, on="id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    valid = merged[
        merged["overall_score"].notna()
    ].copy()

    if len(valid) == 0:
        print("No valid judge results were produced.")
        return

    dimensions = [
        "relevance",
        "groundedness",
        "helpfulness",
        "specificity",
        "safety_escalation",
        "overall_score"
    ]

    print()
    print("=== Reply Quality Results ===")

    for col in dimensions:
        print(
            f"{col}: "
            f"{valid[col].astype(float).mean():.3f}"
        )

    pass_rate = valid["pass"].astype(bool).mean()

    print(f"pass_rate: {pass_rate:.3f}")
    print(f"evaluated: {len(valid)}")

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
