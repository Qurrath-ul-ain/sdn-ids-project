import pandas as pd
from pathlib import Path


DATASET = (
    Path.cwd()
    / "data"
    / "processed"
    / "four_class_dataset.csv"
)


def main():

    print("=" * 70)
    print("FOUR-CLASS DATASET INSPECTION")
    print("=" * 70)

    # Read only the Label column first
    print("\nReading class distribution...")

    label_counts = {}

    for chunk in pd.read_csv(
        DATASET,
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

    print("\nCLASS DISTRIBUTION")
    print("-" * 40)

    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{label:<20} {count:,}"
        )

    # Read only the header
    print("\n" + "=" * 70)
    print("FEATURE COLUMNS")
    print("=" * 70)

    header = pd.read_csv(
        DATASET,
        nrows=0
    )

    print(
        f"\nTotal columns: {len(header.columns)}\n"
    )

    for i, column in enumerate(
        header.columns,
        start=1
    ):

        print(
            f"{i:3}. {column}"
        )


if __name__ == "__main__":
    main()