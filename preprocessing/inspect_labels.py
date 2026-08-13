import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "Downloads" / "archive"

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files.\n")

for file in csv_files:

    print("=" * 70)
    print(file.name)
    print("=" * 70)

    label_counts = {}

    try:
        # Read the file in small chunks instead of loading
        # the entire CSV into RAM.
        for chunk in pd.read_csv(
            file,
            usecols=["Label"],
            chunksize=100_000,
            low_memory=False
        ):

            counts = chunk["Label"].value_counts()

            for label, count in counts.items():

                label = str(label).strip()

                label_counts[label] = (
                    label_counts.get(label, 0)
                    + int(count)
                )

        for label, count in sorted(
            label_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(f"{label}: {count}")

    except Exception as e:

        print("ERROR:", e)

    print()