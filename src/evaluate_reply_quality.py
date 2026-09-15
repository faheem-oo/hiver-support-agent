import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to your .env file."
    )

INPUT_FILE = "outputs/heldout_results.csv"
OUTPUT_FILE = "outputs/llm_reply_quality.csv"

MODEL = "gemini-3.6-flash"

def judge_reply(row, client):
    prompt = f"""
You are evaluating an AI customer-support reply for an AppleSupport
customer-support agent.

Customer message:
{row["text"]}

Predicted intent:
{row["predicted_intent"]}

Escalation decision:
{row["predicted_escalate"]}

Escalation reason:
{row.get("predicted_escalation_reason", "")}

Historical evidence used by the agent:
{row.get("retrieval_evidence", "")}

AI-generated reply:
{row["draft_reply"]}

Evaluate the reply on these dimensions.

1. relevance
Does the reply address the customer's actual problem?

2. groundedness
Does the reply avoid unsupported claims and remain consistent with
the available historical evidence?

3. helpfulness
Does it meaningfully move the customer toward a solution?

4. specificity
Does it provide useful problem-specific guidance rather than being
overly generic?

5. safety_escalation
Does it handle risky, sensitive, unresolved, or severe situations
appropriately and escalate when necessary?

Give each score from 1 to 5.

Return ONLY valid JSON. Do not use markdown fences.

{{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "specificity": 1,
  "safety_escalation": 1,
  "overall": 1,
  "reason": "short explanation"
}}

All scores must be integers from 1 to 5.
"""

    # Retry temporary Gemini failures
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            text = response.text.strip()

            if text.startswith("```"):
                text = (
                    text
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            try:
                return json.loads(text)

            except json.JSONDecodeError:
                return {
                    "relevance": None,
                    "groundedness": None,
                    "helpfulness": None,
                    "specificity": None,
                    "safety_escalation": None,
                    "overall": None,
                    "reason": text,
                }

        except Exception as e:
            error_text = str(e)

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            )

            if not temporary_error:
                raise

            if attempt == 4:
                raise

            wait_time = 10 * (attempt + 1)

            print(
                f"Temporary Gemini error. "
                f"Retry {attempt + 1}/4 in {wait_time} seconds...",
                flush=True
            )

            time.sleep(wait_time)


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} held-out examples.")

    df = df.head(20)
    print(f"Evaluating {len(df)} examples.")


    print(f"Using Gemini model: {MODEL}")
    print("Running LLM reply-quality evaluation...")
    print()

    client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={
        "timeout": 60000
    }
)

    # Load previous successful evaluations if they exist
    if os.path.exists(OUTPUT_FILE):
        previous_df = pd.read_csv(OUTPUT_FILE)

        if "id" in previous_df.columns:
            results = previous_df.to_dict("records")
            completed_ids = set(
                previous_df["id"].astype(int)
            )

            print(
                f"Found {len(results)} completed evaluations."
            )
            print("Resuming from previous progress...")
            print()
        else:
            results = []
            completed_ids = set()
    else:
        results = []
        completed_ids = set()

    for i, row in df.iterrows():

        row_id = int(row["id"])

        # Skip rows already successfully evaluated
        if row_id in completed_ids:
            continue

        print(
            f"Evaluating {i + 1}/{len(df)}...",
            flush=True
        )

        try:
            result = judge_reply(row, client)

            results.append({
                "id": row_id,
                "relevance": result.get("relevance"),
                "groundedness": result.get("groundedness"),
                "helpfulness": result.get("helpfulness"),
                "specificity": result.get("specificity"),
                "safety_escalation": result.get("safety_escalation"),
                "overall": result.get("overall"),
                "reason": result.get("reason", ""),
            })

            completed_ids.add(row_id)

            # Save immediately after every successful evaluation
            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"Completed {len(results)}/{len(df)}",
                flush=True
            )

        except Exception as e:
            print(
                f"ERROR on row {row_id}: {e}",
                flush=True
            )

            print(
                "Stopping safely. Run the script again to resume."
            )

            break

    results_df = pd.DataFrame(results)

    print()
    print("=" * 60)
    print("LLM REPLY-QUALITY RESULTS")
    print("=" * 60)

    for column in [
        "relevance",
        "groundedness",
        "helpfulness",
        "specificity",
        "safety_escalation",
        "overall",
    ]:

        values = pd.to_numeric(
            results_df[column],
            errors="coerce"
        ).dropna()

        if len(values) > 0:
            print(
                f"{column}: "
                f"{values.mean():.2f} / 5"
            )

    print()
    print(f"Evaluated replies: {len(results_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()