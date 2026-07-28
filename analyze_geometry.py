import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from losses import EMOTION_NAMES, build_iemocap_angles, build_target_similarity


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def observed_similarity_matrix(embeddings, labels, num_classes=6):
    embeddings = _normalize_rows(embeddings)
    labels = np.asarray(labels, dtype=np.int64)
    matrix = np.full((num_classes, num_classes), np.nan, dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    sums = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float64)
    for class_id in range(num_classes):
        class_values = embeddings[labels == class_id]
        counts[class_id] = class_values.shape[0]
        if counts[class_id]:
            sums[class_id] = class_values.sum(axis=0)

    for first in range(num_classes):
        for second in range(num_classes):
            if first == second:
                count = counts[first]
                if count > 1:
                    matrix[first, first] = (
                        np.dot(sums[first], sums[first]) - count
                    ) / (count * (count - 1))
            elif counts[first] and counts[second]:
                matrix[first, second] = np.dot(
                    sums[first], sums[second]
                ) / (counts[first] * counts[second])
    return matrix, counts


def compute_geometry_metrics(embeddings, labels):
    raw = np.asarray(embeddings, dtype=np.float64)
    normalized = _normalize_rows(raw)
    observed, counts = observed_similarity_matrix(normalized, labels)
    target = (
        build_target_similarity(build_iemocap_angles())
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    upper = np.triu_indices(6, k=1)
    valid = np.isfinite(observed[upper])
    observed_pairs = observed[upper][valid]
    target_pairs = target[upper][valid]
    error = observed_pairs - target_pairs
    correlation = float("nan")
    if observed_pairs.size > 1:
        if np.std(observed_pairs) > 0 and np.std(target_pairs) > 0:
            correlation = float(
                np.corrcoef(observed_pairs, target_pairs)[0, 1]
            )
    norms = np.linalg.norm(raw, axis=1)
    metrics = {
        "cross_class_mae": (
            float(np.mean(np.abs(error))) if error.size else float("nan")
        ),
        "cross_class_rmse": (
            float(np.sqrt(np.mean(error ** 2)))
            if error.size
            else float("nan")
        ),
        "cross_class_pearson": correlation,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "class_counts": counts.tolist(),
        "within_class_similarity": {
            EMOTION_NAMES[index]: (
                float(observed[index, index])
                if np.isfinite(observed[index, index])
                else None
            )
            for index in range(6)
        },
    }
    return metrics, observed, target, normalized


def _write_matrix(path, matrix):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["emotion"] + EMOTION_NAMES)
        for name, row in zip(EMOTION_NAMES, matrix):
            writer.writerow([name] + [float(value) for value in row])


def _save_heatmap(path, matrix, title, vmin=-1.0, vmax=1.0):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(6))
    axis.set_yticks(range(6))
    axis.set_xticklabels(EMOTION_NAMES, rotation=45, ha="right")
    axis.set_yticklabels(EMOTION_NAMES)
    axis.set_title(title)
    for row in range(6):
        for column in range(6):
            value = matrix[row, column]
            label = "nan" if not np.isfinite(value) else "{:.2f}".format(value)
            axis.text(column, row, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_pca(path, embeddings, labels, max_per_class=500):
    labels = np.asarray(labels, dtype=np.int64)
    selected = []
    for class_id in range(6):
        indices = np.flatnonzero(labels == class_id)[:max_per_class]
        selected.extend(indices.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if selected.size < 2:
        return
    values = embeddings[selected]
    selected_labels = labels[selected]
    points = PCA(n_components=2).fit_transform(values)
    figure, axis = plt.subplots(figsize=(8, 7))
    for class_id, name in enumerate(EMOTION_NAMES):
        class_points = points[selected_labels == class_id]
        if class_points.size:
            axis.scatter(
                class_points[:, 0],
                class_points[:, 1],
                s=12,
                alpha=0.55,
                label=name,
            )
    axis.set_title("PCA of normalized emotion representations")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_geometry_artifacts(
    output_dir,
    prefix,
    embeddings,
    labels,
):
    os.makedirs(output_dir, exist_ok=True)
    metrics, observed, target, normalized = compute_geometry_metrics(
        embeddings, labels
    )
    with open(
        os.path.join(output_dir, "{}_metrics.json".format(prefix)),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(metrics, output, indent=2, allow_nan=True)
    _write_matrix(
        os.path.join(output_dir, "{}_observed.csv".format(prefix)),
        observed,
    )
    _write_matrix(
        os.path.join(output_dir, "{}_target.csv".format(prefix)),
        target,
    )
    absolute_error = np.abs(observed - target)
    _write_matrix(
        os.path.join(output_dir, "{}_absolute_error.csv".format(prefix)),
        absolute_error,
    )
    _save_heatmap(
        os.path.join(output_dir, "{}_observed.png".format(prefix)),
        observed,
        "{} observed cosine similarity".format(prefix),
    )
    _save_heatmap(
        os.path.join(output_dir, "{}_target.png".format(prefix)),
        target,
        "Target circular cosine similarity",
    )
    _save_heatmap(
        os.path.join(output_dir, "{}_absolute_error.png".format(prefix)),
        absolute_error,
        "{} absolute target error".format(prefix),
        vmin=0.0,
        vmax=2.0,
    )
    _save_pca(
        os.path.join(output_dir, "{}_pca.png".format(prefix)),
        normalized,
        labels,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--representation",
        choices=["fusion_features", "embeddings"],
        required=True,
    )
    args = parser.parse_args()
    archive = np.load(args.npz)
    if args.representation not in archive:
        raise KeyError(
            "{} is not present in {}".format(args.representation, args.npz)
        )
    save_geometry_artifacts(
        args.output_dir,
        args.representation,
        archive[args.representation],
        archive["labels"],
    )


if __name__ == "__main__":
    main()

