import pandas as pd
from pathlib import Path

scores = [
    [176, 2, 4, 2, 1, 2],
    [184, 5, 5, 4, 5, 5],
    [2, 4, 4, 3, 3, 5],
    [15, 2, 4, 1, 1, 3],
    [197, 2, 4, 1, 1, 2],
    [177, 2, 4, 1, 1, 3],
    [122, 2, 4, 1, 1, 3],
    [168, 1, 5, 1, 1, 2],
    [194, 3, 3, 2, 3, 3],
    [18, 2, 1, 2, 1, 3],
    [89, 1, 5, 1, 1, 3],
    [93, 5, 4, 4, 5, 5],
    [135, 1, 5, 1, 1, 3],
    [200, 1, 5, 1, 1, 3],
    [60, 5, 5, 5, 5, 5],
    [182, 5, 5, 4, 5, 5],
    [172, 5, 4, 4, 4, 5],
    [38, 2, 5, 1, 1, 3],
    [103, 2, 3, 2, 2, 3],
    [161, 1, 5, 1, 1, 3],
]

columns = [
    "id",
    "relevance",
    "groundedness",
    "helpfulness",
    "specificity",
    "safety_escalation",
]

df = pd.DataFrame(scores, columns=columns)

df["overall_score"] = df[
    [
        "relevance",
        "groundedness",
        "helpfulness",
        "specificity",
        "safety_escalation",
    ]
].mean(axis=1)

df["pass"] = df["overall_score"] >= 4

output = Path("outputs/human_reply_quality.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("=== Human Reply Quality ===")
for column in columns[1:]:
    print(f"{column}: {df[column].mean():.2f} / 5")

print(f"overall_score: {df['overall_score'].mean():.2f} / 5")
print(f"pass_rate: {df['pass'].mean():.2%}")
print(f"evaluated: {len(df)}")
print(f"Saved: {output}")
