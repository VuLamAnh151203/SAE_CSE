"""Visualize SDT-CSE representations with PCA and emotion centroids.

Examples:
    python visualize_multimodal_pca.py \
        --path results/sdt_cse_lambda_0.1/seed_2024/features_test.npz

    python visualize_multimodal_pca.py \
        --path results/sdt_cse_lambda_0.1/seed_2024/features_test.npz \
        --feature-key feature_embedding \
        --dimensions 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "sdt-cse-matplotlib-cache"),
)

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from sklearn.decomposition import PCA


SCRIPT_DIR = Path(__file__).resolve().parent
IEMOCAP_ID2LABEL = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
    5: "frustrated",
}
EMOTION_COLORS = {
    "happy": "#F2C14E",
    "sad": "#4C78A8",
    "neutral": "#8C8C8C",
    "angry": "#D64550",
    "excited": "#F28E2B",
    "frustrated": "#8E5EA2",
}
SDT_CSE_FEATURE_KEYS = (
    "feature_fusion",
    "feature_embedding",
    "feature_l",
    "feature_a",
    "feature_v",
    "feature_l_embedding",
    "feature_a_embedding",
    "feature_v_embedding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project an SDT-CSE NPZ representation to two or three PCA "
            "dimensions, color utterances by emotion, and calculate each "
            "emotion centroid."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help=(
            "Path to features_train.npz, features_valid.npz, or "
            "features_test.npz. Relative paths are also resolved from the "
            "SDT-CSE directory."
        ),
    )
    parser.add_argument(
        "--feature-key",
        choices=SDT_CSE_FEATURE_KEYS,
        default="feature_fusion",
        help="Representation to visualize (default: %(default)s).",
    )
    parser.add_argument(
        "--label-key",
        choices=("labels_emo", "preds_emo"),
        default="labels_emo",
        help="Use gold or predicted emotion IDs (default: %(default)s).",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=(2, 3),
        default=2,
        help="Number of PCA dimensions (default: %(default)s).",
    )
    parser.add_argument(
        "--emotion-names",
        nargs="+",
        help=(
            "Optional emotion names in label-ID order, for example "
            "happy sad neutral angry excited frustrated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory override. By default, outputs are written to "
            "pca_dimension/<condition>/<seed>/."
        ),
    )
    parser.add_argument(
        "--prefix",
        help=(
            "Output prefix (default: <input_stem>_<feature_key>_pca)."
        ),
    )
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved plot resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=16.0,
        help="Scatter point size (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Scatter opacity between 0 and 1 (default: %(default)s).",
    )
    parser.add_argument(
        "--mesh",
        choices=("hull", "none"),
        default="hull",
        help="Draw a convex hull or no boundary (default: %(default)s).",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=0.08,
        help="Convex-hull fill opacity (default: %(default)s).",
    )
    parser.add_argument(
        "--mesh-line-width",
        type=float,
        default=1.8,
        help="Convex-hull outline width (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the saved plot interactively.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    expanded = path.expanduser()
    candidates = []
    if expanded.is_absolute():
        candidates.append(expanded)
    else:
        candidates.extend((Path.cwd() / expanded, SCRIPT_DIR / expanded))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    requested = candidates[-1].resolve()
    suggestion = str(requested)
    suggestion = suggestion.replace(
        "feature_test.npz", "features_test.npz"
    )
    suggestion = suggestion.replace(
        "feature_valid.npz", "features_valid.npz"
    )
    suggestion = suggestion.replace(
        "feature_train.npz", "features_train.npz"
    )
    suggestion = suggestion.replace("_lambda_01", "_lambda_0.1")
    message = "Input NPZ does not exist: {}".format(requested)
    if suggestion != str(requested):
        message += "\nDid you mean: {}?".format(suggestion)
    raise FileNotFoundError(message)


def resolve_output_dir(path: Path | None, input_path: Path) -> Path:
    if path is None:
        seed_name = input_path.parent.name
        condition_name = input_path.parent.parent.name
        return (
            SCRIPT_DIR
            / "pca_dimension"
            / condition_name
            / seed_name
        ).resolve()
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (SCRIPT_DIR / expanded).resolve()


def resolve_emotion_names(
    label_ids: np.ndarray,
    custom_names: Sequence[str] | None,
) -> Dict[int, str]:
    unique_ids = sorted(int(label_id) for label_id in np.unique(label_ids))
    if not unique_ids:
        raise ValueError("No emotion labels are available.")

    if custom_names is not None:
        if min(unique_ids) < 0 or max(unique_ids) >= len(custom_names):
            raise ValueError(
                "--emotion-names must contain an entry for every observed "
                "label ID. Observed IDs: {}".format(unique_ids)
            )
        return {
            label_id: custom_names[label_id] for label_id in unique_ids
        }
    return {
        label_id: IEMOCAP_ID2LABEL.get(
            label_id, "emotion_{}".format(label_id)
        )
        for label_id in unique_ids
    }


def validate_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    feature_key: str,
    label_key: str,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(
            "'{}' must have shape [N, D]; got {}.".format(
                feature_key, features.shape
            )
        )
    labels = labels.reshape(-1)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "'{}' has {} rows but '{}' has {} labels.".format(
                feature_key,
                features.shape[0],
                label_key,
                labels.shape[0],
            )
        )
    if min(features.shape) < dimensions:
        raise ValueError(
            "PCA with {} components requires at least {} samples and "
            "embedding dimensions.".format(dimensions, dimensions)
        )
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("'{}' must be numeric.".format(feature_key))
    if not np.issubdtype(labels.dtype, np.integer):
        try:
            labels = labels.astype(np.int64)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "'{}' must contain integer IDs.".format(label_key)
            ) from error

    finite_rows = np.isfinite(features).all(axis=1)
    if not finite_rows.all():
        removed = int((~finite_rows).sum())
        print(
            "Warning: excluding {} row(s) with NaN or infinity.".format(
                removed
            )
        )
        features = features[finite_rows]
        labels = labels[finite_rows]
    if features.shape[0] < dimensions:
        raise ValueError(
            "Fewer than {} finite embeddings remain.".format(dimensions)
        )
    return (
        features.astype(np.float64, copy=False),
        labels.astype(np.int64, copy=False),
    )


def color_for(emotion: str, index: int) -> str:
    if emotion in EMOTION_COLORS:
        return EMOTION_COLORS[emotion]
    return matplotlib.colors.to_hex(plt.get_cmap("tab10")(index % 10))


def draw_convex_hull_mesh(
    axis: plt.Axes,
    points: np.ndarray,
    color: str,
    fill_alpha: float,
    line_width: float,
) -> bool:
    dimensions = points.shape[1]
    unique_points = np.unique(points, axis=0)
    if dimensions not in (2, 3):
        raise ValueError("Convex hulls support only 2-D or 3-D data.")
    if unique_points.shape[0] < dimensions + 1:
        return False
    try:
        hull = ConvexHull(unique_points)
    except QhullError:
        return False

    outline_alpha = min(1.0, max(0.55, fill_alpha * 6.0))
    if dimensions == 2:
        polygon = unique_points[hull.vertices]
        closed = np.vstack((polygon, polygon[0]))
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=color,
            edgecolor="none",
            alpha=fill_alpha,
            zorder=0,
        )
        axis.plot(
            closed[:, 0],
            closed[:, 1],
            color=color,
            linewidth=line_width,
            alpha=outline_alpha,
            zorder=1,
        )
    else:
        triangles = unique_points[hull.simplices]
        surface = Poly3DCollection(
            triangles,
            facecolors=matplotlib.colors.to_rgba(color, fill_alpha),
            edgecolors=matplotlib.colors.to_rgba(color, outline_alpha),
            linewidths=max(0.25, line_width * 0.3),
        )
        surface.set_zorder(0)
        axis.add_collection3d(surface)
    return True


def write_emotion_means_csv(
    path: Path,
    label_ids: Sequence[int],
    emotion_names: Dict[int, str],
    counts: np.ndarray,
    projected_means: np.ndarray,
    original_means: np.ndarray,
) -> None:
    fieldnames = ["label_id", "emotion", "count"]
    fieldnames.extend(
        "pc{}_mean".format(component + 1)
        for component in range(projected_means.shape[1])
    )
    fieldnames.extend(
        "embedding_mean_{:04d}".format(dimension)
        for dimension in range(original_means.shape[1])
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, label_id in enumerate(label_ids):
            row = {
                "label_id": int(label_id),
                "emotion": emotion_names[int(label_id)],
                "count": int(counts[row_index]),
            }
            row.update(
                {
                    "pc{}_mean".format(component + 1): "{:.10g}".format(
                        projected_means[row_index, component]
                    )
                    for component in range(projected_means.shape[1])
                }
            )
            row.update(
                {
                    "embedding_mean_{:04d}".format(dimension): (
                        "{:.10g}".format(value)
                    )
                    for dimension, value in enumerate(
                        original_means[row_index]
                    )
                }
            )
            writer.writerow(row)


def create_plot(
    output_path: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    label_ids: Sequence[int],
    emotion_names: Dict[int, str],
    counts: np.ndarray,
    projected_means: np.ndarray,
    explained_variance_ratio: np.ndarray,
    title: str,
    point_size: float,
    alpha: float,
    dpi: int,
    legend_title: str,
    mesh: str,
    mesh_alpha: float,
    mesh_line_width: float,
    show: bool,
) -> None:
    dimensions = coordinates.shape[1]
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    if dimensions == 3:
        axis = figure.add_subplot(111, projection="3d")
        axis.view_init(elev=23, azim=-58)
    else:
        axis = figure.add_subplot(111)

    for index, label_id in enumerate(label_ids):
        emotion = emotion_names[int(label_id)]
        color = color_for(emotion, index)
        mask = labels == label_id
        if mesh == "hull":
            draw_convex_hull_mesh(
                axis,
                coordinates[mask],
                color,
                fill_alpha=mesh_alpha,
                line_width=mesh_line_width,
            )
        sample_style = {
            "s": point_size,
            "alpha": alpha,
            "c": color,
            "edgecolors": "none",
            "label": "{} (n={:,})".format(
                emotion.capitalize(), counts[index]
            ),
            "rasterized": True,
        }
        centroid_style = {
            "s": 230,
            "marker": "X",
            "c": color,
            "edgecolors": "black",
            "linewidths": 1.2,
            "zorder": 5,
        }
        if dimensions == 3:
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                coordinates[mask, 2],
                **sample_style,
            )
            axis.scatter(
                projected_means[index, 0],
                projected_means[index, 1],
                projected_means[index, 2],
                **centroid_style,
            )
            axis.text(
                projected_means[index, 0],
                projected_means[index, 1],
                projected_means[index, 2],
                "  {}".format(emotion.capitalize()),
                fontsize=9,
                fontweight="bold",
                color="#202020",
                zorder=6,
            )
        else:
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                **sample_style,
            )
            axis.scatter(
                projected_means[index, 0],
                projected_means[index, 1],
                **centroid_style,
            )
            axis.annotate(
                emotion.capitalize(),
                xy=(
                    projected_means[index, 0],
                    projected_means[index, 1],
                ),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#202020",
                zorder=6,
            )

    axis.set_title(title, fontsize=15, pad=14)
    axis.set_xlabel(
        "PC1 ({:.2f}% variance)".format(
            explained_variance_ratio[0] * 100.0
        ),
        fontsize=11,
    )
    axis.set_ylabel(
        "PC2 ({:.2f}% variance)".format(
            explained_variance_ratio[1] * 100.0
        ),
        fontsize=11,
    )
    if dimensions == 3:
        axis.set_zlabel(
            "PC3 ({:.2f}% variance)".format(
                explained_variance_ratio[2] * 100.0
            ),
            fontsize=11,
            labelpad=8,
        )
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    if dimensions == 2:
        axis.set_axisbelow(True)
    axis.legend(
        title=legend_title,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
    )

    note = (
        "X = emotion centroid  |  shaded boundary = convex hull"
        if mesh == "hull"
        else "X = emotion centroid"
    )
    note_style = {
        "fontsize": 9,
        "color": "#404040",
        "bbox": {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "alpha": 0.88,
        },
    }
    if dimensions == 3:
        axis.text2D(
            0.01, 0.01, note, transform=axis.transAxes, **note_style
        )
    else:
        axis.text(
            0.01, 0.01, note, transform=axis.transAxes, **note_style
        )

    figure.savefig(output_path, dpi=dpi, facecolor="white")
    if show:
        print(
            "Interactive window opened. Close it to finish the command."
        )
        plt.show()
    plt.close(figure)


def validate_options(args: argparse.Namespace) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if not 0.0 <= args.mesh_alpha <= 1.0:
        raise ValueError("--mesh-alpha must be between 0 and 1.")
    if (
        args.dpi <= 0
        or args.point_size <= 0
        or args.mesh_line_width <= 0
    ):
        raise ValueError(
            "--dpi, --point-size, and --mesh-line-width must be positive."
        )


def main() -> None:
    args = parse_args()
    validate_options(args)
    input_path = resolve_input_path(args.path)

    with np.load(input_path, allow_pickle=False) as archive:
        missing = [
            key
            for key in (args.feature_key, args.label_key)
            if key not in archive.files
        ]
        if missing:
            raise KeyError(
                "Missing NPZ key(s): {}. Available keys: {}".format(
                    ", ".join(missing),
                    ", ".join(archive.files),
                )
            )
        features, labels = validate_arrays(
            archive[args.feature_key],
            archive[args.label_key],
            args.feature_key,
            args.label_key,
            args.dimensions,
        )

    emotion_names = resolve_emotion_names(labels, args.emotion_names)
    label_ids = np.asarray(sorted(emotion_names), dtype=np.int64)
    pca = PCA(n_components=args.dimensions)
    coordinates = pca.fit_transform(features)

    counts = np.asarray(
        [(labels == label_id).sum() for label_id in label_ids],
        dtype=np.int64,
    )
    original_means = np.vstack(
        [
            features[labels == label_id].mean(axis=0)
            for label_id in label_ids
        ]
    )
    projected_means = pca.transform(original_means)

    output_dir = resolve_output_dir(args.output_dir, input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_prefix = "{}_{}_pca".format(
        input_path.stem, args.feature_key
    )
    if args.dimensions == 3:
        default_prefix += "_3d"
    prefix = args.prefix or default_prefix
    plot_path = output_dir / "{}.png".format(prefix)
    means_path = output_dir / "{}_emotion_means.csv".format(prefix)
    pca_data_path = output_dir / "{}_data.npz".format(prefix)
    title = args.title or "{}-D PCA of {} by emotion - {}".format(
        args.dimensions,
        args.feature_key,
        input_path.stem,
    )

    create_plot(
        output_path=plot_path,
        coordinates=coordinates,
        labels=labels,
        label_ids=label_ids,
        emotion_names=emotion_names,
        counts=counts,
        projected_means=projected_means,
        explained_variance_ratio=pca.explained_variance_ratio_,
        title=title,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        legend_title=(
            "Predicted emotion"
            if args.label_key == "preds_emo"
            else "Gold emotion"
        ),
        mesh=args.mesh,
        mesh_alpha=args.mesh_alpha,
        mesh_line_width=args.mesh_line_width,
        show=args.show,
    )
    write_emotion_means_csv(
        means_path,
        label_ids,
        emotion_names,
        counts,
        projected_means,
        original_means,
    )
    np.savez_compressed(
        pca_data_path,
        coordinates=coordinates,
        labels=labels,
        label_ids=label_ids,
        emotion_names=np.asarray(
            [emotion_names[int(label_id)] for label_id in label_ids]
        ),
        counts=counts,
        mean_embeddings=original_means,
        mean_embeddings_pca=projected_means,
        pca_components=pca.components_,
        pca_center=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        pca_dimensions=np.asarray(args.dimensions),
        feature_key=np.asarray(args.feature_key),
        label_key=np.asarray(args.label_key),
        source_file=np.asarray(str(input_path)),
    )

    print(
        "Loaded {:,} embeddings with {} dimensions from '{}'.".format(
            features.shape[0],
            features.shape[1],
            args.feature_key,
        )
    )
    component_variance = ", ".join(
        "PC{}={:.2f}%".format(component + 1, ratio * 100.0)
        for component, ratio in enumerate(
            pca.explained_variance_ratio_
        )
    )
    print(
        "PCA explained variance: {}, total={:.2f}%.".format(
            component_variance,
            pca.explained_variance_ratio_.sum() * 100.0,
        )
    )
    print("\nEmotion centroids in PCA space:")
    for index, label_id in enumerate(label_ids):
        means = "  ".join(
            "PC{}={:>9.4f}".format(
                component + 1,
                projected_means[index, component],
            )
            for component in range(args.dimensions)
        )
        print(
            "  {:<12} n={:>4}  {}".format(
                emotion_names[int(label_id)],
                counts[index],
                means,
            )
        )
    print("\nPlot:          {}".format(plot_path))
    print("Emotion means: {}".format(means_path))
    print("PCA data:      {}".format(pca_data_path))


if __name__ == "__main__":
    main()
