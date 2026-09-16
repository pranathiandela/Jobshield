from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_dataset.csv"

# Similarity threshold used to create posting groups.
# 0.90 gives a reasonably strict similarity definition while
# keeping the graph manageable in memory.
SIMILARITY_THRESHOLD = 0.90

N_NEIGHBORS = 10
BATCH_SIZE = 500

N_SPLITS = 5
RANDOM_STATE = 42


def build_similarity_groups(texts):
    """
    Build approximate similarity-connected groups.

    Each posting is connected to its top N nearest neighbors
    when cosine similarity >= SIMILARITY_THRESHOLD.

    The resulting connected components are used as groups so
    highly similar postings are kept together during evaluation.
    """

    print("\nBuilding similarity groups...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)

    print(f"Grouping TF-IDF matrix: {matrix.shape}")

    n_samples = matrix.shape[0]

    neighbors = NearestNeighbors(
        n_neighbors=N_NEIGHBORS,
        metric="cosine",
        n_jobs=1,
    )

    neighbors.fit(matrix)

    # Union-Find structure for connected components.
    parent = np.arange(n_samples, dtype=np.int32)
    size = np.ones(n_samples, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a
        size[root_a] += size[root_b]

    threshold_distance = 1.0 - SIMILARITY_THRESHOLD

    for start in range(0, n_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_samples)

        distances, indices = neighbors.kneighbors(
            matrix[start:end]
        )

        for local_row in range(end - start):
            source = start + local_row

            for distance, neighbor in zip(
                distances[local_row],
                indices[local_row],
            ):
                # Ignore self.
                if source == neighbor:
                    continue

                if distance <= threshold_distance:
                    union(source, int(neighbor))

        print(f"Processed {end}/{n_samples} postings")

    # Compress all components.
    groups = np.empty(n_samples, dtype=np.int32)

    root_to_group = {}
    next_group = 0

    for i in range(n_samples):
        root = find(i)

        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1

        groups[i] = root_to_group[root]

    print(f"Total similarity groups: {next_group}")

    group_sizes = pd.Series(groups).value_counts()

    print(
        f"Multi-posting groups: "
        f"{(group_sizes > 1).sum()}"
    )

    print(
        f"Largest group: "
        f"{group_sizes.max()}"
    )

    return groups


def evaluate_fold(X_train, X_test, y_train, y_test, fold_number):
    """
    Train and evaluate one leakage-safe fold.

    TF-IDF is fitted only on X_train.
    """

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_number}")
    print(f"{'=' * 60}")

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples:     {len(y_test)}")

    print(
        f"Training fraud:   {int(y_train.sum())}"
    )
    print(
        f"Test fraud:       {int(y_test.sum())}"
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Train TF-IDF: {X_train_tfidf.shape}")
    print(f"Test TF-IDF:  {X_test_tfidf.shape}")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_tfidf, y_train)

    probabilities = model.predict_proba(X_test_tfidf)[:, 1]

    predictions = (probabilities >= 0.50).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print("\nMetrics at threshold 0.50:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():
    print("=" * 60)
    print("GROUP-AWARE ML EVALUATION")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    print(f"\nDataset: {DATA_FILE}")
    print(f"Total samples: {len(df)}")

    texts = (
        df["job_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    y = (
        df["fraudulent"]
        .astype(int)
        .to_numpy()
    )

    print("\nClass distribution:")
    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "legitimate",
                1: "fraud",
            }
        )
    )

    # Build similarity groups.
    groups = build_similarity_groups(texts)

    print("\nChecking group label composition...")

    group_df = pd.DataFrame(
        {
            "group": groups,
            "label": y,
        }
    )

    mixed_groups = (
        group_df
        .groupby("group")["label"]
        .nunique()
    )

    mixed_count = int((mixed_groups > 1).sum())

    print(f"Mixed-label groups: {mixed_count}")

    if mixed_count > 0:
        print(
            "Note: some similarity groups contain both "
            "legitimate and fraudulent labels."
        )

    # Group-aware stratified cross-validation.
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(texts, y, groups),
        start=1,
    ):
        X_train = [texts[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_result = evaluate_fold(
            X_train,
            X_test,
            y_train,
            y_test,
            fold_number,
        )

        results.append(fold_result)

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("GROUP-AWARE CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    for metric in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        mean = results_df[metric].mean()
        std = results_df[metric].std(ddof=1)

        print(
            f"{metric.upper():8s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    print("\nPer-fold results:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nEvaluation notes:")
    print(
        "1. Similar postings are grouped before splitting."
    )
    print(
        "2. TF-IDF is fitted separately inside each training fold."
    )
    print(
        "3. The model never sees test-fold text during training."
    )
    print(
        "4. Threshold 0.50 is used only for Precision/Recall/F1."
    )
    print(
        "5. ROC-AUC and PR-AUC use probability scores."
    )
    print(
        "6. Similarity groups are approximate because each posting "
        "is connected to its top nearest neighbors."
    )


if __name__ == "__main__":
    main()
