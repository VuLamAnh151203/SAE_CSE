import os
import sys
import tempfile
import unittest

import numpy as np


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import visualize_multimodal_pca as visualization  # noqa: E402


class MultimodalPCAVisualizationTest(unittest.TestCase):
    def test_default_output_preserves_condition_and_seed(self):
        input_path = (
            visualization.SCRIPT_DIR
            / "results"
            / "sdt_cse_lambda_0.1"
            / "seed_2024"
            / "features_test.npz"
        )
        expected = (
            visualization.SCRIPT_DIR
            / "pca_dimension"
            / "sdt_cse_lambda_0.1"
            / "seed_2024"
        ).resolve()
        self.assertEqual(
            visualization.resolve_output_dir(None, input_path),
            expected,
        )

    def test_cli_generates_plot_centroids_and_pca_archive(self):
        generator = np.random.default_rng(2024)
        labels = np.repeat(np.arange(6, dtype=np.int64), 8)
        features = generator.normal(size=(labels.size, 12)).astype(
            np.float32
        )
        features += labels[:, None] * 0.25

        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "features_test.npz")
            output_dir = os.path.join(directory, "plots")
            np.savez_compressed(
                input_path,
                labels_emo=labels,
                preds_emo=labels,
                feature_fusion=features,
                feature_embedding=features[:, :6],
            )
            previous_argv = sys.argv
            try:
                sys.argv = [
                    "visualize_multimodal_pca.py",
                    "--path",
                    input_path,
                    "--feature-key",
                    "feature_embedding",
                    "--output-dir",
                    output_dir,
                    "--mesh",
                    "none",
                ]
                visualization.main()
            finally:
                sys.argv = previous_argv

            prefix = "features_test_feature_embedding_pca"
            self.assertTrue(
                os.path.isfile(
                    os.path.join(output_dir, "{}.png".format(prefix))
                )
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        output_dir,
                        "{}_emotion_means.csv".format(prefix),
                    )
                )
            )
            data_path = os.path.join(
                output_dir, "{}_data.npz".format(prefix)
            )
            self.assertTrue(os.path.isfile(data_path))
            with np.load(data_path) as archive:
                self.assertEqual(archive["coordinates"].shape, (48, 2))
                self.assertEqual(
                    archive["mean_embeddings_pca"].shape, (6, 2)
                )
                self.assertEqual(
                    str(archive["feature_key"]), "feature_embedding"
                )


if __name__ == "__main__":
    unittest.main()
