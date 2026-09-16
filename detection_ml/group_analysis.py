from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_dataset.csv"


def build_similarity_groups(matrix, threshold):
    """Build similarity groups using batched nearest-neighbor search."""

    n_samples = matrix.shape[0]

    # We only need enough neighbors to identify strong
    # similarity connections. Using 10 instead of 20
    # substantially reduces memory usage.
    n_neighbors = 10

    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        n_jobs=1,
    )

    neighbors.fit(matrix)

    rows = []
    cols = []

    # Process the dataset in batches so that distance/index
    # matrices do not become unnecessarily large.
    batch_size = 500

    for start in range(0, n_samples, batch_size):
        end = min(
            start + batch_size,
            n_samples,
        )

        batch = matrix[start:end]

        distances, indices = neighbors.kneighbors(
            batch
        )

        similarities = 1 - distances

        for local_row in range(len(indices)):
            original_row = start + local_row

            for neighbor_position in range(
                len(indices[local_row])
            ):
                neighbor_index = indices[
                    local_row,
                    neighbor_position,
                ]

                similarity = similarities[
                    local_row,
                    neighbor_position,
                ]

                if (
                    neighbor_index != original_row
                    and similarity >= threshold
                ):
                    rows.append(original_row)
                    cols.append(neighbor_index)

    graph = csr_matrix(
        (
            [1] * len(rows),
            (rows, cols),
        ),
        shape=(n_samples, n_samples),
    )

    # Similarity is treated as an undirected relationship.
    graph = graph.maximum(graph.T)

    number_of_groups, group_labels = connected_components(
        graph,
        directed=False,
    )

    return number_of_groups, group_labels


def analyze_groups(df, group_labels, threshold):
    """Print statistics about similarity groups."""

    group_series = pd.Series(
        group_labels,
        index=df.index,
        name="group",
    )

    group_sizes = group_series.value_counts()

    multi_posting_groups = (
        group_sizes[group_sizes > 1]
    )

    print()
    print("=" * 70)
    print(
        f"GROUP ANALYSIS — THRESHOLD {threshold:.2f}"
    )
    print("=" * 70)

    print()
    print(f"Total groups: {len(group_sizes)}")

    print(
        "Groups containing multiple postings: "
        f"{len(multi_posting_groups)}"
    )

    print(
        "Largest group: "
        f"{group_sizes.max()} postings"
    )

    print(
        "Average group size: "
        f"{group_sizes.mean():.2f}"
    )

    print()
    print("Postings belonging to multi-posting groups: "
          f"{int(multi_posting_groups.sum())}")

    # Determine whether any similarity group contains
    # both legitimate and fraudulent postings.
    group_label_counts = (
        df.assign(group=group_labels)
        .groupby("group")["fraudulent"]
        .nunique()
    )

    mixed_groups = group_label_counts[
        group_label_counts > 1
    ]

    print()
    print(
        "Groups containing BOTH legitimate and fraudulent "
        f"labels: {len(mixed_groups)}"
    )

    if len(mixed_groups) > 0:
        mixed_postings = int(
            group_sizes.loc[
                mixed_groups.index
            ].sum()
        )

        print(
            "Postings inside mixed-label groups: "
            f"{mixed_postings}"
        )

    print()
    print("Largest groups:")

    largest_groups = (
        group_sizes
        .sort_values(ascending=False)
        .head(10)
    )

    for group_id, size in largest_groups.items():

        labels = (
            df.loc[
                group_series == group_id,
                "fraudulent",
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"  Group {group_id}: "
            f"{size} postings, labels={labels}"
        )


def main():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "job_text",
        "fraudulent",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    texts = (
        df["job_text"]
        .fillna("")
        .astype(str)
    )

    print("=" * 70)
    print("MEMORY-EFFICIENT SIMILARITY GROUP ANALYSIS")
    print("=" * 70)

    print()
    print(f"Total postings: {len(df)}")

    print()
    print("Building TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
    )

    matrix = vectorizer.fit_transform(texts)

    print(
        f"TF-IDF matrix: {matrix.shape}"
    )

    for threshold in [
        0.95,
        0.90,
        0.85,
    ]:

        print()
        print(
            f"Building groups at threshold "
            f"{threshold:.2f}..."
        )

        number_of_groups, group_labels = (
            build_similarity_groups(
                matrix,
                threshold,
            )
        )

        analyze_groups(
            df,
            group_labels,
            threshold,
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "No dataset rows were modified."
    )

    print(
        "No model was modified."
    )

    print(
        "No Flask files were modified."
    )


if __name__ == "__main__":
    main()
