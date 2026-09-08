import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path.home() / "Downloads" / "archive"

OUTPUT_DIR = (
    Path.cwd()
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "four_class_dataset.csv"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAP = {

    # Benign
    "Benign": "Benign",

    # Brute Force
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",

    # Botnet
    "Bot": "Botnet",

    # Web Attacks
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
}


# ============================================================
# FILES WE NEED
# ============================================================

FILES = [

    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",

    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",

    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",

    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
]


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    first_file = True

    total_rows = 0

    for filename in FILES:

        file_path = DATA_DIR / filename

        if not file_path.exists():

            print(
                f"ERROR: File not found: {file_path}"
            )

            continue

        print("\n" + "=" * 70)

        print(
            f"Processing: {filename}"
        )

        print("=" * 70)

        # Process in chunks to avoid RAM problems
        for chunk in pd.read_csv(
            file_path,
            chunksize=100_000,
            low_memory=False
        ):

            # Remove accidental whitespace
            chunk["Label"] = (
                chunk["Label"]
                .astype(str)
                .str.strip()
            )

            # Keep only our required classes
            chunk = chunk[
                chunk["Label"].isin(
                    LABEL_MAP.keys()
                )
            ].copy()

            if chunk.empty:
                continue

            # Map labels
            chunk["Label"] = (
                chunk["Label"]
                .map(LABEL_MAP)
            )

            # Append to output
            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_file else "a",
                header=first_file,
                index=False
            )

            first_file = False

            total_rows += len(chunk)

            print(
                f"Rows collected: {total_rows:,}"
            )

    print("\n" + "=" * 70)

    print("DATASET CREATION COMPLETE")

    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Total rows: {total_rows:,}"
    )


if __name__ == "__main__":
    main()