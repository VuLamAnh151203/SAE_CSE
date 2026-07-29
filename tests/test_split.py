import os
import sys
import unittest

import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataloader import (  # noqa: E402
    fixed_train_validation_test_split,
    original_test_selection_split,
)
from aggregate_results import (  # noqa: E402
    condition_name,
    residual_gate_statistics,
)
from train import (  # noqa: E402
    build_argument_parser,
    confusion_pair_metrics,
    experiment_directory_name,
    is_better_selection,
    is_better_validation,
    validate_arguments,
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

    def test_validation_selection_tie_break(self):
        self.assertTrue(is_better_validation(70.0, 1.0, 69.0, 0.5))
        self.assertTrue(is_better_validation(70.0, 0.4, 70.0, 0.5))
        self.assertFalse(is_better_validation(70.0, 0.6, 70.0, 0.5))

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

        fixed = parser.parse_args(
            ["--experiment-mode", "sdt_cse"]
        )
        validate_arguments(fixed)
        self.assertEqual(fixed.circular_geometry, "equal")
        self.assertEqual(fixed.angle_weight, 0.0)
        self.assertEqual(fixed.confusion_gap_weight, 0.0)
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
        labels = np.asarray([0, 4, 0, 4, 3, 5, 3, 5])
        predictions = np.asarray([0, 4, 4, 0, 3, 5, 5, 3])
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


if __name__ == "__main__":
    unittest.main()
