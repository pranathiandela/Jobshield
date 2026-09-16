import html
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "DataSet.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_dataset.csv"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

REQUIRED_COLUMNS = TEXT_COLUMNS + [
    "fraudulent",
]


def clean_text(value):
    """Convert a raw job-posting field into normalized plain text."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Decode HTML entities such as &amp;, &nbsp;, etc.
    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset():
    """Load and validate the raw Detection dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def prepare_dataset(df):
    """Clean text, labels, and duplicates for ML training."""

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Clean the target label
    # ---------------------------------------------------------

    df["fraudulent"] = (
        df["fraudulent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "f": 0,
            "t": 1,
        })
    )

    # Remove rows with an invalid target label.
    df = df.dropna(subset=["fraudulent"])

    df["fraudulent"] = df["fraudulent"].astype(int)

    # ---------------------------------------------------------
    # 2. Clean all text fields
    # ---------------------------------------------------------

    for column in TEXT_COLUMNS:
        df[column] = df[column].apply(clean_text)

    # ---------------------------------------------------------
    # 3. Create one combined text field
    # ---------------------------------------------------------

    df["job_text"] = (
        "TITLE: " + df["title"] + " "
        + "COMPANY: " + df["company_profile"] + " "
        + "DESCRIPTION: " + df["description"] + " "
        + "REQUIREMENTS: " + df["requirements"] + " "
        + "BENEFITS: " + df["benefits"]
    ).str.strip()

    # ---------------------------------------------------------
    # 4. Remove completely empty postings
    # ---------------------------------------------------------

    df = df[df["job_text"].str.len() > 0].copy()

    # ---------------------------------------------------------
    # 5. Remove exact duplicate rows
    # ---------------------------------------------------------

    before_exact = len(df)

    df = df.drop_duplicates().copy()

    removed_exact = before_exact - len(df)

    # ---------------------------------------------------------
    # 6. Remove duplicate combined job postings
    #
    # This is important for avoiding train/test leakage.
    # Keep the first occurrence only.
    # ---------------------------------------------------------

    before_text = len(df)

    df = df.drop_duplicates(
        subset=["job_text"],
        keep="first",
    ).copy()

    removed_text = before_text - len(df)

    # ---------------------------------------------------------
    # 7. Keep only columns needed for training/evaluation.
    #
    # in_balanced_dataset is intentionally excluded.
    # ---------------------------------------------------------

    result = df[
        [
            "title",
            "job_text",
            "fraudulent",
        ]
    ].reset_index(drop=True)

    print("Preprocessing complete.")
    print()
    print(f"Original rows: {len(pd.read_csv(INPUT_FILE))}")
    print(f"Removed exact duplicates: {removed_exact}")
    print(f"Removed duplicate job texts: {removed_text}")
    print(f"Final rows: {len(result)}")
    print()

    print("Class distribution:")
    print(result["fraudulent"].value_counts().sort_index())
    print()

    print("Class percentages:")
    print(
        (
            result["fraudulent"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )
    print()

    return result


def main():
    df = load_dataset()
    cleaned = prepare_dataset(df)

    cleaned.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Cleaned dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
