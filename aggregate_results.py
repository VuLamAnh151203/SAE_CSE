import argparse
import csv
import glob
import json
import os

import numpy as np


METRICS = (
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "macro_recall",
    "fusion_ce",
    "circular_cse",
    "angle_regularization",
)


def load_summaries(output_dir):
    paths = glob.glob(
        os.path.join(output_dir, "*", "seed_*", "summary.json")
    )
    summaries = []
    for path in sorted(paths):
        with open(path, encoding="utf-8") as source:
            summary = json.load(source)
        summary["_path"] = path
        summaries.append(summary)
    return summaries


def condition_name(summary):
    mode = summary["experiment_mode"]
    if mode == "sdt_cse_learnable_angles":
        geometry = summary.get("circular_geometry", "nrc_vad")
        condition = "{}_{}_lambda_{}_angle_{}".format(
            mode,
            geometry,
            format(float(summary["circular_weight"]), "g"),
            format(float(summary.get("angle_weight", 0.1)), "g"),
        )
    elif mode in (
        "sdt_cse",
        "sdt_cse_all_cosine",
        "sdt_cse_fusion_only",
    ):
        geometry = summary.get("circular_geometry", "equal")
        geometry_suffix = (
            "" if geometry == "equal"
            else "_{}".format(geometry)
        )
        condition = "{}{}_lambda_{}".format(
            mode,
            geometry_suffix,
            format(float(summary["circular_weight"]), "g"),
        )
    else:
        condition = mode
    if summary.get("sdt_residual_update", "standard") == "spherical":
        condition += "_spherical_residual_a{}_m{}".format(
            format(
                float(
                    summary.get(
                        "spherical_attention_alpha_init", 0.1
                    )
                ),
                "g",
            ),
            format(
                float(
                    summary.get("spherical_mlp_alpha_init", 0.1)
                ),
                "g",
            ),
        )
    if summary.get("selection_protocol", "validation") == "test":
        condition += "_test_selected"
    return condition


def residual_gate_statistics(summary):
    gates = summary.get("spherical_residual_gates") or {}
    attention = np.asarray(
        [
            value
            for name, value in gates.items()
            if name.endswith("attention_update")
        ],
        dtype=np.float64,
    )
    mlp = np.asarray(
        [
            value
            for name, value in gates.items()
            if name.endswith("mlp_update")
        ],
        dtype=np.float64,
    )
    all_values = np.concatenate(
        [values for values in (attention, mlp) if values.size]
    ) if attention.size or mlp.size else np.asarray([], dtype=np.float64)
    return {
        "final_attention_alpha_mean": (
            float(attention.mean()) if attention.size else None
        ),
        "final_mlp_alpha_mean": (
            float(mlp.mean()) if mlp.size else None
        ),
        "final_alpha_min": (
            float(all_values.min()) if all_values.size else None
        ),
        "final_alpha_max": (
            float(all_values.max()) if all_values.size else None
        ),
    }


def aggregate(output_dir):
    summaries = load_summaries(output_dir)
    if not summaries:
        raise FileNotFoundError(
            "no seed summaries found below {}".format(output_dir)
        )
    aggregate_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    rows = []
    grouped = {}
    for summary in summaries:
        condition = condition_name(summary)
        row = {
            "condition": condition,
            "seed": summary["seed"],
            "selected_epoch": summary["selected_epoch"],
            "sdt_residual_update": summary.get(
                "sdt_residual_update", "standard"
            ),
        }
        for metric in METRICS:
            row["test_{}".format(metric)] = summary["test"].get(metric)
        row.update(residual_gate_statistics(summary))
        rows.append(row)
        grouped.setdefault(condition, []).append(row)

    per_seed_path = os.path.join(aggregate_dir, "per_seed.csv")
    with open(per_seed_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for condition, condition_rows in sorted(grouped.items()):
        item = {
            "condition": condition,
            "runs": len(condition_rows),
        }
        for metric in METRICS:
            values = np.asarray(
                [
                    row["test_{}".format(metric)]
                    for row in condition_rows
                    if row["test_{}".format(metric)] is not None
                ],
                dtype=np.float64,
            )
            item["{}_mean".format(metric)] = (
                float(values.mean()) if values.size else ""
            )
            item["{}_std".format(metric)] = (
                float(values.std()) if values.size else ""
            )
        for metric in (
            "final_attention_alpha_mean",
            "final_mlp_alpha_mean",
            "final_alpha_min",
            "final_alpha_max",
        ):
            values = np.asarray(
                [
                    row[metric]
                    for row in condition_rows
                    if row[metric] is not None
                ],
                dtype=np.float64,
            )
            item["{}_mean".format(metric)] = (
                float(values.mean()) if values.size else ""
            )
            item["{}_std".format(metric)] = (
                float(values.std()) if values.size else ""
            )
        aggregate_rows.append(item)
    aggregate_path = os.path.join(aggregate_dir, "summary.csv")
    with open(aggregate_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(aggregate_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    paired_rows = []
    by_condition_seed = {
        (row["condition"], int(row["seed"])): row for row in rows
    }
    conditions = sorted(grouped)
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1 :]:
            common_seeds = sorted(
                {
                    int(row["seed"]) for row in grouped[left]
                }
                & {
                    int(row["seed"]) for row in grouped[right]
                }
            )
            for seed in common_seeds:
                item = {
                    "left_condition": left,
                    "right_condition": right,
                    "seed": seed,
                }
                for metric in METRICS:
                    left_value = by_condition_seed[(left, seed)][
                        "test_{}".format(metric)
                    ]
                    right_value = by_condition_seed[(right, seed)][
                        "test_{}".format(metric)
                    ]
                    item["right_minus_left_{}".format(metric)] = (
                        None
                        if left_value is None or right_value is None
                        else right_value - left_value
                    )
                paired_rows.append(item)
    if paired_rows:
        with open(
            os.path.join(aggregate_dir, "paired_differences.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as output:
            writer = csv.DictWriter(
                output, fieldnames=list(paired_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(paired_rows)
    return aggregate_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(aggregate(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
