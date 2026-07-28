import os
import sys
import unittest


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataloader import fixed_train_validation_test_split  # noqa: E402
from train import (  # noqa: E402
    build_argument_parser,
    experiment_directory_name,
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

    def test_learnable_mode_uses_nrc_default_and_fixed_modes_do_not(self):
        parser = build_argument_parser()
        learnable = parser.parse_args(
            ["--experiment-mode", "sdt_cse_learnable_angles"]
        )
        validate_arguments(learnable)
        self.assertEqual(learnable.circular_geometry, "nrc_vad")
        self.assertEqual(learnable.angle_weight, 0.1)

        fixed = parser.parse_args(
            ["--experiment-mode", "sdt_cse"]
        )
        validate_arguments(fixed)
        self.assertEqual(fixed.circular_geometry, "equal")
        self.assertEqual(fixed.angle_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
