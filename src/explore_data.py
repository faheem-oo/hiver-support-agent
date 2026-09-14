import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/archive/twcs/twcs.csv")

print("Loading dataset...")
print(f"File: {DATA_PATH}")

if not DATA_PATH.exists():
    print("\nERROR: twcs.csv was not found!")
    print("Expected location:")
    print(DATA_PATH.resolve())
    exit()

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)

print(df.head().to_string())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)

print("\nDone!")