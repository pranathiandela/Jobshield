from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_dataset.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    texts = df["job_text"].fillna("").astype(str)
    labels = df["fraudulent"].astype(int).values

    print("=" * 70)
    print("NEAR-DUPLICATE / SIMILARITY CHECK")
    print("=" * 70)

    print()
    print(f"Total postings: {len(df)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
    )

    matrix = vectorizer.fit_transform(texts)

    neighbors = NearestNeighbors(
        n_neighbors=2,
        metric="cosine",
        n_jobs=-1,
    )

    neighbors.fit(matrix)

    distances, indices = neighbors.kneighbors(matrix)

    similarities = 1 - distances[:, 1]
    nearest_indices = indices[:, 1]

    results = pd.DataFrame({
        "index": range(len(df)),
        "nearest_index": nearest_indices,
        "similarity": similarities,
        "label": labels,
        "nearest_label": labels[nearest_indices],
    })

    print()
    print("Similarity statistics:")
    print(
        results["similarity"].describe().round(4)
    )

    print()
    print("=" * 70)
    print("HIGH-SIMILARITY PAIRS")
    print("=" * 70)

    high_similarity = results[
        results["similarity"] >= 0.85
    ].sort_values(
        "similarity",
        ascending=False,
    )

    print()
    print(
        f"Pairs with similarity >= 0.85: "
        f"{len(high_similarity)}"
    )

    print()
    print(
        f"{'Similarity':<14}"
        f"{'Label':<10}"
        f"{'Nearest Label':<16}"
    )

    print("-" * 70)

    for _, row in high_similarity.head(20).iterrows():
        print(
            f"{row['similarity']:<14.4f}"
            f"{row['label']:<10}"
            f"{row['nearest_label']:<16}"
        )

    print()
    print("=" * 70)
    print("CROSS-LABEL HIGH-SIMILARITY PAIRS")
    print("=" * 70)

    cross_label = results[
        (results["similarity"] >= 0.85)
        & (results["label"] != results["nearest_label"])
    ].sort_values(
        "similarity",
        ascending=False,
    )

    print()
    print(
        f"Cross-label pairs with similarity >= 0.85: "
        f"{len(cross_label)}"
    )

    for i, (_, row) in enumerate(
        cross_label.head(10).iterrows(),
        start=1,
    ):
        current_index = int(row["index"])
        nearest_index = int(row["nearest_index"])

        print()
        print(f"--- Pair #{i} ---")
        print(
            f"Similarity: {row['similarity']:.4f}"
        )
        print(
            f"Labels: "
            f"{labels[current_index]} -> "
            f"{labels[nearest_index]}"
        )

        print()
        print("POSTING A:")
        print(texts.iloc[current_index][:800])

        print()
        print("POSTING B:")
        print(texts.iloc[nearest_index][:800])


if __name__ == "__main__":
    main()
