import os
import sys
import unittest


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataloader import fixed_train_validation_test_split  # noqa: E402
from train import is_better_validation  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

