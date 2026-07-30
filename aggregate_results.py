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
    "text_circular_cse",
    "audio_circular_cse",
    "visual_circular_cse",
    "unimodal_circular_cse",
    "total_circular_cse",
    "angle_regularization",
    "gap_prior_regularization",
    "confusion_gap_regularization",
    "confusion_classification_margin",
    "hypo_compactness",
    "hypo_dispersion",
    "hypo_alignment",
    "hypo_total",
    "happy_excited_pair_macro_f1",
    "happy_excited_pair_weighted_f1",
    "happy_excited_mutual_confusion_rate",
    "angry_frustrated_pair_macro_f1",
    "angry_frustrated_pair_weighted_f1",
    "angry_frustrated_mutual_confusion_rate",
    "sad_neutral_pair_macro_f1",
    "sad_neutral_pair_weighted_f1",
    "sad_neutral_mutual_confusion_rate",
)
ORDERED_GAP_NAMES = (
    "happy_to_excited",
    "excited_to_angry",
    "angry_to_frustrated",
    "frustrated_to_sad",
    "sad_to_neutral",
    "neutral_to_happy",
)
CONFUSION_GAP_NAMES = (
    "happy_excited",
    "angry_frustrated",
    "sad_neutral",
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
    if mode == "sdt_hypo":
        condition = "{}_lambda_{}_w{}_tau{}_pm{}".format(
            mode,
            format(
                float(summary.get("hypo_loss_weight", 0.1)), "g"
            ),
            format(
                float(
                    summary.get("hypo_compactness_weight", 2.0)
                ),
                "g",
            ),
            format(
                float(summary.get("hypo_temperature", 0.1)), "g"
            ),
            format(
                float(
                    summary.get("hypo_prototype_momentum", 0.95)
                ),
                "g",
            ),
        )
    elif mode in (
        "sdt_cse_bilevel_all_gaps",
        "sdt_cse_bilevel_all_gaps_train_holdout",
    ):
        geometry = summary.get("bilevel_geometry") or {}
        if mode == "sdt_cse_bilevel_all_gaps_train_holdout":
            condition = (
                "{}_l{}_i{}_mg{}_pr{}_p{}_cm{}_cw{}_alr{}_oc{}_ah{}"
            ).format(
                mode,
                format(float(summary["circular_weight"]), "g"),
                geometry.get("initialization", "equal"),
                format(
                    float(
                        geometry.get(
                            "minimum_class_gap_degrees", 20.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get("gap_prior_weight", 0.01)
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confused_cse_pair_weight", 5.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_margin", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_weight", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "angle_learning_rate", 0.001
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "outer_confusion_weight", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "angle_holdout_ratio",
                            summary.get("angle_holdout_ratio", 0.1),
                        )
                    ),
                    "g",
                ),
            )
        else:
            condition = (
                "{}_lambda_{}_init_{}_mingap_{}_prior_{}_pair_{}_"
                "clsmargin_{}_clsweight_{}"
            ).format(
                mode,
                format(float(summary["circular_weight"]), "g"),
                geometry.get("initialization", "equal"),
                format(
                    float(
                        geometry.get(
                            "minimum_class_gap_degrees", 20.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get("gap_prior_weight", 0.01)
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confused_cse_pair_weight", 5.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_margin", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_weight", 0.1
                        )
                    ),
                    "g",
                ),
            )
            learning_source = geometry.get(
                "learning_source",
                summary.get(
                    "all_gap_learning_source", "validation"
                ),
            )
            if learning_source == "validation":
                condition += "_anglelr_{}_outerconf_{}".format(
                    format(
                        float(
                            geometry.get(
                                "angle_learning_rate", 0.001
                            )
                        ),
                        "g",
                    ),
                    format(
                        float(
                            geometry.get(
                                "outer_confusion_weight", 0.1
                            )
                        ),
                        "g",
                    ),
                )
            else:
                condition += "_source_{}".format(learning_source)
    elif mode == "sdt_cse_bilevel_confusion_gap_hypo_aligned":
        geometry = summary.get("bilevel_geometry") or {}
        condition = (
            "{}_lambda_{}_range_{}-{}_init_{}_hlambda_{}_"
            "w{}_a{}_tau{}_pm{}"
        ).format(
            mode,
            format(float(summary["circular_weight"]), "g"),
            format(
                float(geometry.get("minimum_degrees", 70.0)), "g"
            ),
            format(
                float(geometry.get("maximum_degrees", 110.0)), "g"
            ),
            format(
                float(geometry.get("initial_degrees", 90.0)), "g"
            ),
            format(
                float(summary.get("hypo_loss_weight", 0.1)), "g"
            ),
            format(
                float(
                    summary.get("hypo_compactness_weight", 2.0)
                ),
                "g",
            ),
            format(
                float(summary.get("hypo_alignment_weight", 1.0)),
                "g",
            ),
            format(
                float(summary.get("hypo_temperature", 0.1)), "g"
            ),
            format(
                float(
                    summary.get("hypo_prototype_momentum", 0.95)
                ),
                "g",
            ),
        )
    elif mode in (
        "sdt_cse_bilevel_confusion_gap",
        "sdt_cse_bilevel_confusion_gap_train_holdout",
    ):
        geometry = summary.get("bilevel_geometry") or {}
        if mode == "sdt_cse_bilevel_confusion_gap_train_holdout":
            condition = (
                "{}_l{}_r{}-{}_i{}_p{}_cm{}_cw{}_alr{}_oc{}_ah{}"
            ).format(
                mode,
                format(float(summary["circular_weight"]), "g"),
                format(
                    float(geometry.get("minimum_degrees", 70.0)),
                    "g",
                ),
                format(
                    float(geometry.get("maximum_degrees", 110.0)),
                    "g",
                ),
                format(
                    float(geometry.get("initial_degrees", 90.0)),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confused_cse_pair_weight", 5.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_margin", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_weight", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "angle_learning_rate", 0.001
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "outer_confusion_weight", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "angle_holdout_ratio",
                            summary.get("angle_holdout_ratio", 0.1),
                        )
                    ),
                    "g",
                ),
            )
        else:
            condition = (
                "{}_lambda_{}_range_{}-{}_init_{}_pair_{}_"
                "clsmargin_{}_clsweight_{}_anglelr_{}_outerconf_{}"
            ).format(
                mode,
                format(float(summary["circular_weight"]), "g"),
                format(
                    float(geometry.get("minimum_degrees", 70.0)),
                    "g",
                ),
                format(
                    float(geometry.get("maximum_degrees", 110.0)),
                    "g",
                ),
                format(
                    float(geometry.get("initial_degrees", 90.0)),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confused_cse_pair_weight", 5.0
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_margin", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        summary.get(
                            "confusion_classification_weight", 0.1
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "angle_learning_rate", 0.001
                        )
                    ),
                    "g",
                ),
                format(
                    float(
                        geometry.get(
                            "outer_confusion_weight", 0.1
                        )
                    ),
                    "g",
                ),
            )
    elif mode == "sdt_cse_confusion_margin":
        condition = (
            "{}_{}_lambda_{}_mingap_{}_pair_{}_"
            "clsmargin_{}_clsweight_{}"
        ).format(
            mode,
            summary.get(
                "circular_geometry", "confusion_separated"
            ),
            format(float(summary["circular_weight"]), "g"),
            format(
                float(
                    summary.get(
                        "minimum_confusion_gap_degrees", 75.0
                    )
                ),
                "g",
            ),
            format(
                float(
                    summary.get("confused_cse_pair_weight", 5.0)
                ),
                "g",
            ),
            format(
                float(
                    summary.get(
                        "confusion_classification_margin", 0.1
                    )
                ),
                "g",
            ),
            format(
                float(
                    summary.get(
                        "confusion_classification_weight", 0.1
                    )
                ),
                "g",
            ),
        )
    elif (
        mode
        == "sdt_cse_learnable_angles_confusion_gap_hypo_aligned"
    ):
        condition = (
            "{}_{}_l{}_ang{}_g{}_gw{}_hl{}_c{}_a{}_t{}_"
            "pm{}_wu{}_r{}"
        ).format(
            mode,
            summary.get("circular_geometry", "equal"),
            format(float(summary["circular_weight"]), "g"),
            format(float(summary.get("angle_weight", 0.1)), "g"),
            format(
                float(
                    summary.get(
                        "minimum_confusion_gap_degrees", 75.0
                    )
                ),
                "g",
            ),
            format(
                float(summary.get("confusion_gap_weight", 0.1)),
                "g",
            ),
            format(
                float(summary.get("hypo_loss_weight", 0.02)), "g"
            ),
            format(
                float(
                    summary.get("hypo_compactness_weight", 1.0)
                ),
                "g",
            ),
            format(
                float(summary.get("hypo_alignment_weight", 0.1)),
                "g",
            ),
            format(
                float(summary.get("hypo_temperature", 0.2)), "g"
            ),
            format(
                float(
                    summary.get("hypo_prototype_momentum", 0.9)
                ),
                "g",
            ),
            int(summary.get("hypo_warmup_epochs", 10)),
            int(summary.get("hypo_ramp_epochs", 20)),
        )
    elif mode in (
        "sdt_cse_learnable_angles_confusion_gap",
        "sdt_cse_learnable_angles_confusion_gap_sad_neutral",
    ):
        geometry = summary.get("circular_geometry", "equal")
        condition = (
            "{}_{}_lambda_{}_angle_{}_mingap_{}_gap_{}"
        ).format(
            mode,
            geometry,
            format(float(summary["circular_weight"]), "g"),
            format(float(summary.get("angle_weight", 0.1)), "g"),
            format(
                float(
                    summary.get(
                        "minimum_confusion_gap_degrees", 75.0
                    )
                ),
                "g",
            ),
            format(
                float(summary.get("confusion_gap_weight", 0.1)),
                "g",
            ),
        )
    elif mode == "sdt_cse_learnable_angles":
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
        "sdt_cse_all_modal_cse",
        "sdt_cse_fusion_only",
        "sdt_cse_confusion_margin",
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
        bilevel_geometry = summary.get("bilevel_geometry") or {}
        named_gaps = bilevel_geometry.get(
            "named_gaps_degrees", {}
        )
        confusion_pair_gaps = summary.get(
            "confusion_pair_gaps_degrees"
        ) or {}
        row = {
            "condition": condition,
            "seed": summary["seed"],
            "selected_epoch": summary["selected_epoch"],
            "angle_learning_split": summary.get(
                "angle_learning_split"
            ),
            "angle_holdout_ratio": summary.get(
                "angle_holdout_ratio"
            ),
            "sdt_residual_update": summary.get(
                "sdt_residual_update", "standard"
            ),
            "hypo_loss_weight": summary.get("hypo_loss_weight"),
            "hypo_compactness_weight": summary.get(
                "hypo_compactness_weight"
            ),
            "hypo_temperature": summary.get("hypo_temperature"),
            "hypo_prototype_momentum": summary.get(
                "hypo_prototype_momentum"
            ),
            "hypo_alignment_weight": summary.get(
                "hypo_alignment_weight"
            ),
            "hypo_warmup_epochs": summary.get(
                "hypo_warmup_epochs"
            ),
            "hypo_ramp_epochs": summary.get(
                "hypo_ramp_epochs"
            ),
            "selected_hypo_schedule_scale": summary.get(
                "selected_hypo_schedule_scale"
            ),
            "hypo_alignment_target_detached": summary.get(
                "hypo_alignment_target_detached"
            ),
            "hypo_initialized_classes": summary.get(
                "hypo_initialized_classes"
            ),
            "hypo_prototype_coverage": summary.get(
                "hypo_prototype_coverage"
            ),
            "selected_bilevel_gap_degrees": (
                bilevel_geometry.get(
                    "selected_confusion_gap_degrees"
                )
            ),
            "selected_bilevel_minimum_gap_degrees": (
                bilevel_geometry.get(
                    "selected_minimum_gap_degrees"
                )
            ),
            "selected_bilevel_maximum_gap_degrees": (
                bilevel_geometry.get(
                    "selected_maximum_gap_degrees"
                )
            ),
            "selected_bilevel_gap_prior_regularization": (
                bilevel_geometry.get(
                    "gap_prior_regularization"
                )
            ),
        }
        for gap_name in ORDERED_GAP_NAMES:
            row[
                "selected_gap_{}_degrees".format(gap_name)
            ] = named_gaps.get(gap_name)
        for pair_name in CONFUSION_GAP_NAMES:
            row[
                "selected_confusion_gap_{}_degrees".format(
                    pair_name
                )
            ] = confusion_pair_gaps.get(pair_name)
        validation = summary.get("validation") or {}
        angle_holdout = summary.get("angle_holdout") or {}
        for metric in METRICS:
            row["test_{}".format(metric)] = summary["test"].get(metric)
            row["validation_{}".format(metric)] = validation.get(
                metric
            )
            row["angle_holdout_{}".format(metric)] = (
                angle_holdout.get(metric)
            )
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
            "angle_learning_split": condition_rows[0].get(
                "angle_learning_split"
            ),
            "angle_holdout_ratio": condition_rows[0].get(
                "angle_holdout_ratio"
            ),
            "hypo_loss_weight": condition_rows[0].get(
                "hypo_loss_weight"
            ),
            "hypo_compactness_weight": condition_rows[0].get(
                "hypo_compactness_weight"
            ),
            "hypo_temperature": condition_rows[0].get(
                "hypo_temperature"
            ),
            "hypo_prototype_momentum": condition_rows[0].get(
                "hypo_prototype_momentum"
            ),
            "hypo_alignment_weight": condition_rows[0].get(
                "hypo_alignment_weight"
            ),
            "hypo_warmup_epochs": condition_rows[0].get(
                "hypo_warmup_epochs"
            ),
            "hypo_ramp_epochs": condition_rows[0].get(
                "hypo_ramp_epochs"
            ),
            "hypo_alignment_target_detached": condition_rows[
                0
            ].get("hypo_alignment_target_detached"),
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
            validation_values = np.asarray(
                [
                    row["validation_{}".format(metric)]
                    for row in condition_rows
                    if row[
                        "validation_{}".format(metric)
                    ] is not None
                ],
                dtype=np.float64,
            )
            item["validation_{}_mean".format(metric)] = (
                float(validation_values.mean())
                if validation_values.size
                else ""
            )
            item["validation_{}_std".format(metric)] = (
                float(validation_values.std())
                if validation_values.size
                else ""
            )
            angle_holdout_values = np.asarray(
                [
                    row["angle_holdout_{}".format(metric)]
                    for row in condition_rows
                    if row[
                        "angle_holdout_{}".format(metric)
                    ] is not None
                ],
                dtype=np.float64,
            )
            item["angle_holdout_{}_mean".format(metric)] = (
                float(angle_holdout_values.mean())
                if angle_holdout_values.size
                else ""
            )
            item["angle_holdout_{}_std".format(metric)] = (
                float(angle_holdout_values.std())
                if angle_holdout_values.size
                else ""
            )
        for metric in (
            "final_attention_alpha_mean",
            "final_mlp_alpha_mean",
            "final_alpha_min",
            "final_alpha_max",
            "selected_bilevel_gap_degrees",
            "selected_bilevel_minimum_gap_degrees",
            "selected_bilevel_maximum_gap_degrees",
            "selected_bilevel_gap_prior_regularization",
            "hypo_initialized_classes",
            "hypo_prototype_coverage",
            *(
                "selected_gap_{}_degrees".format(name)
                for name in ORDERED_GAP_NAMES
            ),
            *(
                "selected_confusion_gap_{}_degrees".format(name)
                for name in CONFUSION_GAP_NAMES
            ),
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
                    left_validation = by_condition_seed[(left, seed)][
                        "validation_{}".format(metric)
                    ]
                    right_validation = by_condition_seed[(right, seed)][
                        "validation_{}".format(metric)
                    ]
                    item[
                        "validation_right_minus_left_{}".format(
                            metric
                        )
                    ] = (
                        None
                        if (
                            left_validation is None
                            or right_validation is None
                        )
                        else right_validation - left_validation
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
