import csv
import os
import sys
import tempfile
import unittest

import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataloader import (  # noqa: E402
    fixed_train_angle_validation_test_split,
    fixed_train_validation_test_split,
    original_test_selection_split,
)
from aggregate_results import (  # noqa: E402
    condition_name,
    residual_gate_statistics,
)
from train import (  # noqa: E402
    bilevel_outer_split_name,
    build_argument_parser,
    confusion_pair_metrics,
    experiment_directory_name,
    is_better_selection,
    is_better_validation,
    uses_bilevel_gap_learning,
    uses_validation_gap_learning,
    validate_arguments,
    write_epoch_metrics,
)


class FixedSplitTest(unittest.TestCase):
    def test_first_ten_percent_is_validation(self):
        train_ids = ["train_{:02d}".format(i) for i in range(20)]
        test_ids = ["test_0", "test_1"]
        splits = fixed_train_validation_test_split(
            train_ids, test_ids, validation_ratio=0.10
        )
        self.assertEqual(splits["validation"], train_ids[:2])
        self.assertEqual(splits["training"], train_ids[2:])
        self.assertEqual(splits["testing"], test_ids)
        self.assertFalse(
            set(splits["training"]) & set(splits["testing"])
        )

    def test_invalid_or_overlapping_splits_fail(self):
        with self.assertRaises(ValueError):
            fixed_train_validation_test_split(
                ["a", "b"], ["a"], validation_ratio=0.5
            )
        with self.assertRaises(ValueError):
            fixed_train_validation_test_split(
                ["a"], ["b"], validation_ratio=0.1
            )

    def test_angle_holdout_split_is_disjoint_80_10_10(self):
        train_ids = ["train_{:02d}".format(i) for i in range(20)]
        test_ids = ["test_0", "test_1"]
        splits = fixed_train_angle_validation_test_split(
            train_ids,
            test_ids,
            validation_ratio=0.10,
            angle_holdout_ratio=0.10,
        )
        self.assertEqual(splits["validation"], train_ids[:2])
        self.assertEqual(splits["angle_holdout"], train_ids[2:4])
        self.assertEqual(splits["training"], train_ids[4:])
        self.assertEqual(splits["testing"], test_ids)
        names = (
            "training",
            "angle_holdout",
            "validation",
            "testing",
        )
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                self.assertFalse(
                    set(splits[first]) & set(splits[second])
                )
        self.assertEqual(
            set(splits["training"])
            | set(splits["angle_holdout"])
            | set(splits["validation"]),
            set(train_ids),
        )
        with self.assertRaises(ValueError):
            fixed_train_angle_validation_test_split(
                train_ids,
                test_ids,
                validation_ratio=0.5,
                angle_holdout_ratio=0.5,
            )
        with self.assertRaises(ValueError):
            fixed_train_angle_validation_test_split(
                train_ids,
                ["train_00"],
                validation_ratio=0.1,
                angle_holdout_ratio=0.1,
            )

    def test_validation_selection_tie_break(self):
        self.assertTrue(is_better_validation(70.0, 1.0, 69.0, 0.5))
        self.assertTrue(is_better_validation(70.0, 0.4, 70.0, 0.5))
        self.assertFalse(is_better_validation(70.0, 0.6, 70.0, 0.5))

    def test_epoch_csv_includes_angle_holdout_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "epoch_metrics.csv")
            write_epoch_metrics(
                path,
                [
                    {
                        "epoch": 1,
                        "seconds": 0.5,
                        "training": {"weighted_f1": 60.0},
                        "angle_holdout": {"weighted_f1": 61.0},
                        "validation": {"weighted_f1": 62.0},
                    }
                ],
            )
            with open(path, newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
        self.assertEqual(
            float(row["angle_holdout_weighted_f1"]), 61.0
        )
        self.assertEqual(float(row["validation_weighted_f1"]), 62.0)

    def test_original_test_protocol_uses_all_training_dialogues(self):
        train_ids = ["train_0", "train_1", "train_2"]
        test_ids = ["test_0", "test_1"]
        splits = original_test_selection_split(train_ids, test_ids)
        self.assertEqual(splits["training"], train_ids)
        self.assertEqual(splits["validation"], [])
        self.assertEqual(splits["testing"], test_ids)
        self.assertFalse(
            set(splits["training"]) & set(splits["testing"])
        )

    def test_test_selection_uses_strict_f1_and_keeps_earlier_ties(self):
        self.assertTrue(
            is_better_selection(
                70.0, 2.0, 69.0, 1.0, selection_protocol="test"
            )
        )
        self.assertFalse(
            is_better_selection(
                70.0, 0.1, 70.0, 1.0, selection_protocol="test"
            )
        )
        self.assertFalse(
            is_better_selection(
                70.004,
                0.1,
                70.001,
                1.0,
                selection_protocol="test",
            )
        )

    def test_nonuniform_geometry_has_an_independent_run_directory(self):
        self.assertEqual(
            experiment_directory_name("sdt_cse", 0.1, "equal"),
            "sdt_cse_lambda_0.1",
        )
        self.assertEqual(
            experiment_directory_name("sdt_cse", 0.1, "nrc_vad"),
            "sdt_cse_nrc_vad_lambda_0.1",
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_all_modal_cse",
                0.1,
                "equal",
            ),
            "sdt_cse_all_modal_cse_lambda_0.1",
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_learnable_angles",
                0.1,
                "nrc_vad",
                0.1,
            ),
            (
                "sdt_cse_learnable_angles_nrc_vad_"
                "lambda_0.1_angle_0.1"
            ),
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_learnable_angles_confusion_gap",
                0.1,
                "equal",
                0.1,
                "validation",
                "standard",
                0.1,
                0.1,
                75.0,
                0.1,
            ),
            (
                "sdt_cse_learnable_angles_confusion_gap_equal_"
                "lambda_0.1_angle_0.1_mingap_75_gap_0.1"
            ),
        )
        three_pair_mode = (
            "sdt_cse_learnable_angles_confusion_gap_sad_neutral"
        )
        three_pair_condition = experiment_directory_name(
            three_pair_mode,
            0.1,
            "equal",
            angle_weight=0.1,
            minimum_confusion_gap_degrees=75.0,
            confusion_gap_weight=0.1,
        )
        self.assertEqual(
            three_pair_condition,
            (
                "{}_equal_lambda_0.1_angle_0.1_mingap_75_gap_0.1"
            ).format(three_pair_mode),
        )
        self.assertEqual(
            condition_name(
                {
                    "experiment_mode": three_pair_mode,
                    "circular_geometry": "equal",
                    "circular_weight": 0.1,
                    "angle_weight": 0.1,
                    "minimum_confusion_gap_degrees": 75.0,
                    "confusion_gap_weight": 0.1,
                }
            ),
            three_pair_condition,
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_confusion_margin",
                0.1,
                "confusion_separated",
                0.0,
                "validation",
                "standard",
                0.1,
                0.1,
                75.0,
                0.0,
                5.0,
                0.1,
                0.1,
            ),
            (
                "sdt_cse_confusion_margin_confusion_separated_"
                "lambda_0.1_mingap_75_pair_5_clsmargin_0.1_"
                "clsweight_0.1"
            ),
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_bilevel_confusion_gap",
                0.1,
                "confusion_separated",
                0.0,
                "validation",
                "standard",
                0.1,
                0.1,
                70.0,
                0.0,
                5.0,
                0.1,
                0.1,
                70.0,
                110.0,
                90.0,
                0.001,
                0.1,
            ),
            (
                "sdt_cse_bilevel_confusion_gap_lambda_0.1_"
                "range_70-110_init_90_pair_5_clsmargin_0.1_"
                "clsweight_0.1_anglelr_0.001_outerconf_0.1"
            ),
        )
        holdout_condition = experiment_directory_name(
            "sdt_cse_bilevel_confusion_gap_train_holdout",
            0.1,
            "confusion_separated",
            confused_cse_pair_weight=5.0,
            confusion_classification_margin=0.1,
            confusion_classification_weight=0.1,
            bilevel_gap_minimum_degrees=70.0,
            bilevel_gap_maximum_degrees=110.0,
            bilevel_gap_initial_degrees=90.0,
            bilevel_angle_learning_rate=0.001,
            bilevel_outer_confusion_weight=0.1,
            angle_holdout_ratio=0.1,
        )
        self.assertEqual(
            holdout_condition,
            (
                "sdt_cse_bilevel_confusion_gap_train_holdout_"
                "l0.1_r70-110_i90_p5_cm0.1_cw0.1_alr0.001_"
                "oc0.1_ah0.1"
            ),
        )
        self.assertEqual(
            condition_name(
                {
                    "experiment_mode": (
                        "sdt_cse_bilevel_confusion_gap_train_holdout"
                    ),
                    "circular_weight": 0.1,
                    "selection_protocol": "validation",
                    "confused_cse_pair_weight": 5.0,
                    "confusion_classification_margin": 0.1,
                    "confusion_classification_weight": 0.1,
                    "angle_holdout_ratio": 0.1,
                    "bilevel_geometry": {
                        "minimum_degrees": 70.0,
                        "maximum_degrees": 110.0,
                        "initial_degrees": 90.0,
                        "angle_learning_rate": 0.001,
                        "outer_confusion_weight": 0.1,
                        "angle_holdout_ratio": 0.1,
                    },
                }
            ),
            holdout_condition,
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_bilevel_all_gaps",
                0.1,
                "equal",
                confused_cse_pair_weight=5.0,
                confusion_classification_margin=0.1,
                confusion_classification_weight=0.1,
                bilevel_angle_learning_rate=0.001,
                bilevel_outer_confusion_weight=0.1,
                bilevel_all_gaps_initialization="equal",
                bilevel_minimum_class_gap_degrees=20.0,
                bilevel_gap_prior_weight=0.01,
            ),
            (
                "sdt_cse_bilevel_all_gaps_lambda_0.1_init_equal_"
                "mingap_20_prior_0.01_pair_5_clsmargin_0.1_"
                "clsweight_0.1_anglelr_0.001_outerconf_0.1"
            ),
        )
        all_gap_holdout_condition = experiment_directory_name(
            "sdt_cse_bilevel_all_gaps_train_holdout",
            0.1,
            "equal",
            confused_cse_pair_weight=5.0,
            confusion_classification_margin=0.1,
            confusion_classification_weight=0.1,
            bilevel_angle_learning_rate=0.001,
            bilevel_outer_confusion_weight=0.1,
            bilevel_all_gaps_initialization="equal",
            bilevel_minimum_class_gap_degrees=20.0,
            bilevel_gap_prior_weight=0.01,
            angle_holdout_ratio=0.1,
        )
        self.assertEqual(
            all_gap_holdout_condition,
            (
                "sdt_cse_bilevel_all_gaps_train_holdout_l0.1_"
                "iequal_mg20_pr0.01_p5_cm0.1_cw0.1_alr0.001_"
                "oc0.1_ah0.1"
            ),
        )
        self.assertEqual(
            condition_name(
                {
                    "experiment_mode": (
                        "sdt_cse_bilevel_all_gaps_train_holdout"
                    ),
                    "circular_weight": 0.1,
                    "selection_protocol": "validation",
                    "confused_cse_pair_weight": 5.0,
                    "confusion_classification_margin": 0.1,
                    "confusion_classification_weight": 0.1,
                    "angle_holdout_ratio": 0.1,
                    "bilevel_geometry": {
                        "initialization": "equal",
                        "minimum_class_gap_degrees": 20.0,
                        "gap_prior_weight": 0.01,
                        "angle_learning_rate": 0.001,
                        "outer_confusion_weight": 0.1,
                        "angle_holdout_ratio": 0.1,
                    },
                }
            ),
            all_gap_holdout_condition,
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_bilevel_all_gaps",
                0.1,
                "equal",
                confused_cse_pair_weight=5.0,
                confusion_classification_margin=0.1,
                confusion_classification_weight=0.1,
                bilevel_all_gaps_initialization="equal",
                bilevel_minimum_class_gap_degrees=20.0,
                bilevel_gap_prior_weight=0.01,
                all_gap_learning_source="training",
            ),
            (
                "sdt_cse_bilevel_all_gaps_lambda_0.1_init_equal_"
                "mingap_20_prior_0.01_pair_5_clsmargin_0.1_"
                "clsweight_0.1_source_training"
            ),
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse",
                0.1,
                "equal",
                0.0,
                "test",
            ),
            "sdt_cse_lambda_0.1_test_selected",
        )
        self.assertEqual(
            experiment_directory_name(
                "sdt_cse_learnable_angles",
                0.1,
                "equal",
                0.1,
                "test",
                "spherical",
                0.1,
                0.2,
            ),
            (
                "sdt_cse_learnable_angles_equal_lambda_0.1_"
                "angle_0.1_spherical_residual_a0.1_m0.2_"
                "test_selected"
            ),
        )

    def test_learnable_mode_uses_nrc_default_and_fixed_modes_do_not(self):
        parser = build_argument_parser()
        learnable = parser.parse_args(
            ["--experiment-mode", "sdt_cse_learnable_angles"]
        )
        validate_arguments(learnable)
        self.assertEqual(learnable.circular_geometry, "nrc_vad")
        self.assertEqual(learnable.angle_weight, 0.1)

        confusion_gap = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_learnable_angles_confusion_gap",
            ]
        )
        validate_arguments(confusion_gap)
        self.assertEqual(confusion_gap.circular_geometry, "equal")
        self.assertEqual(confusion_gap.angle_weight, 0.1)
        self.assertEqual(confusion_gap.confusion_gap_weight, 0.1)
        self.assertEqual(
            confusion_gap.minimum_confusion_gap_degrees, 75.0
        )

        invalid_prior = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_learnable_angles_confusion_gap",
                "--circular-geometry",
                "nrc_vad",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_prior)

        confusion_margin = parser.parse_args(
            ["--experiment-mode", "sdt_cse_confusion_margin"]
        )
        validate_arguments(confusion_margin)
        self.assertEqual(
            confusion_margin.circular_geometry,
            "confusion_separated",
        )
        self.assertEqual(
            confusion_margin.confused_cse_pair_weight, 5.0
        )
        self.assertEqual(
            confusion_margin.confusion_classification_margin, 0.1
        )
        self.assertEqual(
            confusion_margin.confusion_classification_weight, 0.1
        )
        self.assertEqual(confusion_margin.angle_weight, 0.0)
        self.assertEqual(confusion_margin.confusion_gap_weight, 0.0)

        bilevel = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_confusion_gap",
            ]
        )
        validate_arguments(bilevel)
        self.assertEqual(
            bilevel.circular_geometry, "confusion_separated"
        )
        self.assertEqual(bilevel.bilevel_gap_minimum_degrees, 70.0)
        self.assertEqual(bilevel.bilevel_gap_initial_degrees, 90.0)
        self.assertEqual(bilevel.bilevel_gap_maximum_degrees, 110.0)
        self.assertEqual(bilevel.bilevel_inner_step_size, bilevel.lr)
        self.assertEqual(bilevel.angle_weight, 0.0)
        self.assertEqual(bilevel.confusion_gap_weight, 0.0)
        self.assertEqual(bilevel.selection_protocol, "validation")

        invalid_bilevel = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_confusion_gap",
                "--selection-protocol",
                "test",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_bilevel)

        train_holdout = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_confusion_gap_train_holdout",
            ]
        )
        validate_arguments(train_holdout)
        self.assertEqual(train_holdout.validation_ratio, 0.1)
        self.assertEqual(train_holdout.angle_holdout_ratio, 0.1)
        self.assertEqual(
            train_holdout.selection_protocol, "validation"
        )
        self.assertFalse(
            uses_validation_gap_learning(train_holdout)
        )
        self.assertTrue(uses_bilevel_gap_learning(train_holdout))
        self.assertEqual(
            bilevel_outer_split_name(train_holdout),
            "angle_holdout",
        )

        invalid_holdout_selection = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_confusion_gap_train_holdout",
                "--selection-protocol",
                "test",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_holdout_selection)

        invalid_holdout_sum = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_confusion_gap_train_holdout",
                "--validation-ratio",
                "0.6",
                "--angle-holdout-ratio",
                "0.4",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_holdout_sum)

        invalid_holdout_mode = parser.parse_args(
            [
                "--experiment-mode",
                "sdt",
                "--angle-holdout-ratio",
                "0.1",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_holdout_mode)

        all_gaps = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps",
            ]
        )
        validate_arguments(all_gaps)
        self.assertEqual(all_gaps.circular_geometry, "equal")
        self.assertEqual(
            all_gaps.bilevel_all_gaps_initialization, "equal"
        )
        self.assertEqual(
            all_gaps.bilevel_minimum_class_gap_degrees, 20.0
        )
        self.assertEqual(all_gaps.bilevel_gap_prior_weight, 0.01)
        self.assertEqual(all_gaps.selection_protocol, "validation")
        self.assertEqual(all_gaps.confused_cse_pair_weight, 5.0)
        self.assertEqual(
            all_gaps.confusion_classification_weight, 0.1
        )
        self.assertEqual(
            all_gaps.all_gap_learning_source, "validation"
        )

        all_gaps_holdout = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps_train_holdout",
            ]
        )
        validate_arguments(all_gaps_holdout)
        self.assertEqual(
            all_gaps_holdout.circular_geometry, "equal"
        )
        self.assertEqual(
            all_gaps_holdout.angle_holdout_ratio, 0.1
        )
        self.assertFalse(
            uses_validation_gap_learning(all_gaps_holdout)
        )
        self.assertTrue(
            uses_bilevel_gap_learning(all_gaps_holdout)
        )
        self.assertEqual(
            bilevel_outer_split_name(all_gaps_holdout),
            "angle_holdout",
        )

        invalid_all_gaps_holdout_source = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps_train_holdout",
                "--all-gap-learning-source",
                "training",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_all_gaps_holdout_source)

        nrc_all_gaps = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps",
                "--bilevel-all-gaps-initialization",
                "nrc_vad",
                "--bilevel-minimum-class-gap-degrees",
                "5",
            ]
        )
        validate_arguments(nrc_all_gaps)
        self.assertEqual(nrc_all_gaps.circular_geometry, "nrc_vad")

        invalid_nrc_floor = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps",
                "--bilevel-all-gaps-initialization",
                "nrc_vad",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_nrc_floor)

        invalid_all_gaps = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps",
                "--selection-protocol",
                "test",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_all_gaps)

        training_all_gaps = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_cse_bilevel_all_gaps",
                "--all-gap-learning-source",
                "training",
                "--selection-protocol",
                "test",
            ]
        )
        validate_arguments(training_all_gaps)
        self.assertEqual(
            training_all_gaps.all_gap_learning_source, "training"
        )
        self.assertEqual(
            training_all_gaps.selection_protocol, "test"
        )

        invalid_source_mode = parser.parse_args(
            [
                "--experiment-mode",
                "sdt",
                "--all-gap-learning-source",
                "training",
            ]
        )
        with self.assertRaises(ValueError):
            validate_arguments(invalid_source_mode)

        fixed = parser.parse_args(
            ["--experiment-mode", "sdt_cse"]
        )
        validate_arguments(fixed)
        self.assertEqual(fixed.circular_geometry, "equal")
        self.assertEqual(fixed.angle_weight, 0.0)
        self.assertEqual(fixed.confusion_gap_weight, 0.0)
        self.assertEqual(fixed.confused_cse_pair_weight, 1.0)
        self.assertEqual(
            fixed.confusion_classification_weight, 0.0
        )
        self.assertEqual(fixed.sdt_residual_update, "standard")

    def test_aggregation_separates_test_selected_results(self):
        summary = {
            "experiment_mode": "sdt_cse",
            "circular_geometry": "equal",
            "circular_weight": 0.1,
            "selection_protocol": "test",
        }
        self.assertEqual(
            condition_name(summary),
            "sdt_cse_lambda_0.1_test_selected",
        )
        all_modal = dict(summary)
        all_modal["experiment_mode"] = "sdt_cse_all_modal_cse"
        self.assertEqual(
            condition_name(all_modal),
            "sdt_cse_all_modal_cse_lambda_0.1_test_selected",
        )
        confusion_gap = dict(summary)
        confusion_gap.update(
            {
                "experiment_mode": (
                    "sdt_cse_learnable_angles_confusion_gap"
                ),
                "angle_weight": 0.1,
                "minimum_confusion_gap_degrees": 75.0,
                "confusion_gap_weight": 1.0,
            }
        )
        self.assertEqual(
            condition_name(confusion_gap),
            (
                "sdt_cse_learnable_angles_confusion_gap_equal_"
                "lambda_0.1_angle_0.1_mingap_75_gap_1_"
                "test_selected"
            ),
        )
        confusion_margin = dict(summary)
        confusion_margin.update(
            {
                "experiment_mode": "sdt_cse_confusion_margin",
                "circular_geometry": "confusion_separated",
                "minimum_confusion_gap_degrees": 75.0,
                "confused_cse_pair_weight": 5.0,
                "confusion_classification_margin": 0.1,
                "confusion_classification_weight": 0.1,
            }
        )
        self.assertEqual(
            condition_name(confusion_margin),
            (
                "sdt_cse_confusion_margin_confusion_separated_"
                "lambda_0.1_mingap_75_pair_5_clsmargin_0.1_"
                "clsweight_0.1_test_selected"
            ),
        )
        bilevel = dict(summary)
        bilevel.update(
            {
                "experiment_mode": "sdt_cse_bilevel_confusion_gap",
                "selection_protocol": "validation",
                "bilevel_geometry": {
                    "minimum_degrees": 70.0,
                    "maximum_degrees": 110.0,
                    "initial_degrees": 90.0,
                    "angle_learning_rate": 0.001,
                    "outer_confusion_weight": 0.1,
                },
                "confused_cse_pair_weight": 5.0,
                "confusion_classification_margin": 0.1,
                "confusion_classification_weight": 0.1,
            }
        )
        self.assertEqual(
            condition_name(bilevel),
            (
                "sdt_cse_bilevel_confusion_gap_lambda_0.1_"
                "range_70-110_init_90_pair_5_clsmargin_0.1_"
                "clsweight_0.1_anglelr_0.001_outerconf_0.1"
            ),
        )
        all_gaps = dict(summary)
        all_gaps.update(
            {
                "experiment_mode": "sdt_cse_bilevel_all_gaps",
                "selection_protocol": "validation",
                "bilevel_geometry": {
                    "initialization": "equal",
                    "minimum_class_gap_degrees": 20.0,
                    "gap_prior_weight": 0.01,
                    "angle_learning_rate": 0.001,
                    "outer_confusion_weight": 0.1,
                },
                "confused_cse_pair_weight": 5.0,
                "confusion_classification_margin": 0.1,
                "confusion_classification_weight": 0.1,
            }
        )
        self.assertEqual(
            condition_name(all_gaps),
            (
                "sdt_cse_bilevel_all_gaps_lambda_0.1_init_equal_"
                "mingap_20_prior_0.01_pair_5_clsmargin_0.1_"
                "clsweight_0.1_anglelr_0.001_outerconf_0.1"
            ),
        )
        all_gaps_holdout = dict(all_gaps)
        all_gaps_holdout.update(
            {
                "experiment_mode": (
                    "sdt_cse_bilevel_all_gaps_train_holdout"
                ),
                "angle_holdout_ratio": 0.1,
                "bilevel_geometry": dict(
                    all_gaps["bilevel_geometry"],
                    learning_source="angle_holdout",
                    angle_holdout_ratio=0.1,
                ),
            }
        )
        self.assertEqual(
            condition_name(all_gaps_holdout),
            (
                "sdt_cse_bilevel_all_gaps_train_holdout_l0.1_"
                "iequal_mg20_pr0.01_p5_cm0.1_cw0.1_alr0.001_"
                "oc0.1_ah0.1"
            ),
        )
        training_all_gaps = dict(all_gaps)
        training_all_gaps["bilevel_geometry"] = dict(
            all_gaps["bilevel_geometry"],
            learning_source="training",
        )
        self.assertEqual(
            condition_name(training_all_gaps),
            (
                "sdt_cse_bilevel_all_gaps_lambda_0.1_init_equal_"
                "mingap_20_prior_0.01_pair_5_clsmargin_0.1_"
                "clsweight_0.1_source_training"
            ),
        )

        spherical = dict(summary)
        spherical.update(
            {
                "sdt_residual_update": "spherical",
                "spherical_attention_alpha_init": 0.1,
                "spherical_mlp_alpha_init": 0.2,
            }
        )
        self.assertEqual(
            condition_name(spherical),
            (
                "sdt_cse_lambda_0.1_spherical_residual_"
                "a0.1_m0.2_test_selected"
            ),
        )
        spherical["spherical_residual_gates"] = {
            "t_t.transformer_inter.0.attention_update": 0.11,
            "a_t.transformer_inter.0.attention_update": 0.13,
            "t_t.transformer_inter.0.mlp_update": 0.21,
            "a_t.transformer_inter.0.mlp_update": 0.23,
        }
        statistics = residual_gate_statistics(spherical)
        self.assertAlmostEqual(
            statistics["final_attention_alpha_mean"], 0.12
        )
        self.assertAlmostEqual(
            statistics["final_mlp_alpha_mean"], 0.22
        )
        self.assertAlmostEqual(statistics["final_alpha_min"], 0.11)
        self.assertAlmostEqual(statistics["final_alpha_max"], 0.23)

    def test_confusion_pair_metrics_are_reported_separately(self):
        labels = np.asarray(
            [0, 4, 0, 4, 3, 5, 3, 5, 1, 2, 1, 2]
        )
        predictions = np.asarray(
            [0, 4, 4, 0, 3, 5, 5, 3, 1, 2, 2, 1]
        )
        metrics = confusion_pair_metrics(labels, predictions)
        self.assertAlmostEqual(
            metrics["happy_excited_pair_macro_f1"], 50.0
        )
        self.assertAlmostEqual(
            metrics["happy_excited_mutual_confusion_rate"], 50.0
        )
        self.assertAlmostEqual(
            metrics["angry_frustrated_pair_macro_f1"], 50.0
        )
        self.assertAlmostEqual(
            metrics["angry_frustrated_mutual_confusion_rate"],
            50.0,
        )
        self.assertAlmostEqual(
            metrics["sad_neutral_pair_macro_f1"], 50.0
        )
        self.assertAlmostEqual(
            metrics["sad_neutral_mutual_confusion_rate"], 50.0
        )


if __name__ == "__main__":
    unittest.main()
