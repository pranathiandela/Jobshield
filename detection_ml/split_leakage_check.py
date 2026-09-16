from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_dataset.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df["job_text"].fillna("").astype(str)
    y = df["fraudulent"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("=" * 70)
    print("TRAIN / TEST SIMILARITY LEAKAGE CHECK")
    print("=" * 70)

    print()
    print(f"Training postings: {len(X_train)}")
    print(f"Testing postings : {len(X_test)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Find the closest training posting for every test posting.
    neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        n_jobs=-1,
    )

    neighbors.fit(X_train_tfidf)

    distances, indices = neighbors.kneighbors(
        X_test_tfidf
    )

    similarities = 1 - distances[:, 0]
    nearest_train_indices = indices[:, 0]

    train_labels = y_train.to_numpy()
    test_labels = y_test.to_numpy()

    nearest_train_labels = train_labels[
        nearest_train_indices
    ]

    results = pd.DataFrame({
        "test_index": range(len(X_test)),
        "train_index": nearest_train_indices,
        "similarity": similarities,
        "test_label": test_labels,
        "train_label": nearest_train_labels,
    })

    print()
    print("Nearest train/test similarity statistics:")
    print(
        results["similarity"]
        .describe()
        .round(4)
    )

    print()
    print("=" * 70)
    print("SIMILARITY THRESHOLDS")
    print("=" * 70)

    for threshold in [0.95, 0.90, 0.85, 0.80, 0.75]:

        matching = results[
            results["similarity"] >= threshold
        ]

        same_label = matching[
            matching["test_label"]
            == matching["train_label"]
        ]

        different_label = matching[
            matching["test_label"]
            != matching["train_label"]
        ]

        print()
        print(
            f"Threshold >= {threshold:.2f}"
        )
        print(
            f"Similar test postings : "
            f"{len(matching)}"
        )
        print(
            f"Same-label pairs      : "
            f"{len(same_label)}"
        )
        print(
            f"Different-label pairs : "
            f"{len(different_label)}"
        )

    print()
    print("=" * 70)
    print("TOP TRAIN / TEST SIMILAR PAIRS")
    print("=" * 70)

    top_pairs = results.sort_values(
        "similarity",
        ascending=False,
    ).head(15)

    for i, (_, row) in enumerate(
        top_pairs.iterrows(),
        start=1,
    ):
        test_position = int(row["test_index"])
        train_position = int(row["train_index"])

        test_text = X_test.iloc[test_position]
        train_text = X_train.iloc[train_position]

        print()
        print(f"--- Pair #{i} ---")
        print(
            f"Similarity : "
            f"{row['similarity']:.4f}"
        )
        print(
            f"Test label : "
            f"{int(row['test_label'])}"
        )
        print(
            f"Train label: "
            f"{int(row['train_label'])}"
        )

        print()
        print("TEST POSTING:")
        print(test_text[:700])

        print()
        print("TRAIN POSTING:")
        print(train_text[:700])


if __name__ == "__main__":
    main()
