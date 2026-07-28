import os
import sys
import tempfile
import unittest

import numpy as np


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from train import emotion_to_sentiment, save_feature_npz  # noqa: E402


class FeatureExportTest(unittest.TestCase):
    def _result(self, include_embedding=True):
        result = {
            "labels_array": np.asarray([0, 3], dtype=np.int64),
            "predictions_array": np.asarray([2, 4], dtype=np.int64),
            "prediction_rows": [
                {
                    "video_id": "Ses01F_impro01",
                    "utterance_index": 0,
                    "utterance_id": "Ses01F_impro01_F000",
                    "sentence": "First sentence.",
                },
                {
                    "video_id": "Ses01F_impro01",
                    "utterance_index": 1,
                    "utterance_id": "Ses01F_impro01_M001",
                    "sentence": "Second sentence.",
                },
            ],
            "text_features_array": np.ones((2, 4), dtype=np.float32),
            "visual_features_array": np.ones((2, 4), dtype=np.float32) * 2,
            "audio_features_array": np.ones((2, 4), dtype=np.float32) * 3,
            "fusion_features_array": np.ones((2, 4), dtype=np.float32) * 4,
            "embeddings_array": (
                np.ones((2, 3), dtype=np.float32)
                if include_embedding
                else None
            ),
        }
        return result

    def test_sentiment_mapping_matches_iemocap_contract(self):
        mapped = emotion_to_sentiment(np.arange(6))
        np.testing.assert_array_equal(mapped, [2, 0, 1, 0, 2, 0])

    def test_npz_matches_reference_keys_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "features_train.npz")
            save_feature_npz(path, self._result(include_embedding=True))
            with np.load(path) as archive:
                expected = {
                    "labels_emo",
                    "preds_emo",
                    "labels_sen",
                    "preds_sen",
                    "dialogue_ids",
                    "utterance_indices",
                    "utterance_ids",
                    "sentences",
                    "feature_l",
                    "feature_v",
                    "feature_a",
                    "feature_fusion",
                    "feature_embedding",
                }
                self.assertEqual(set(archive.files), expected)
                self.assertEqual(archive["labels_emo"].dtype, np.int64)
                self.assertEqual(
                    archive["feature_fusion"].dtype, np.float32
                )
                self.assertEqual(
                    archive["feature_fusion"].shape, (2, 4)
                )
                self.assertEqual(
                    archive["feature_embedding"].shape, (2, 3)
                )
                self.assertEqual(
                    archive["utterance_ids"][1],
                    "Ses01F_impro01_M001",
                )
                self.assertEqual(
                    archive["sentences"][0], "First sentence."
                )

    def test_linear_sdt_export_omits_projected_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "features_test.npz")
            save_feature_npz(path, self._result(include_embedding=False))
            with np.load(path) as archive:
                self.assertNotIn("feature_embedding", archive.files)
                self.assertIn("feature_fusion", archive.files)


if __name__ == "__main__":
    unittest.main()
