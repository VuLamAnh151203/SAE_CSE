import math
import os
import sys
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from losses import (  # noqa: E402
    CircularCSELoss,
    build_iemocap_angles,
    build_target_similarity,
    compute_sdt_cse_losses,
    iemocap_class_weights,
    masked_self_distillation_kl,
)


class CircularCSELossTest(unittest.TestCase):
    def test_target_similarity_matches_requested_matrix(self):
        target = build_target_similarity(build_iemocap_angles())
        expected = torch.tensor(
            [
                [1, -0.5, 0.5, -0.5, 0.5, -1],
                [-0.5, 1, 0.5, -0.5, -1, 0.5],
                [0.5, 0.5, 1, -1, -0.5, -0.5],
                [-0.5, -0.5, -1, 1, 0.5, 0.5],
                [0.5, -1, -0.5, 0.5, 1, -0.5],
                [-1, 0.5, -0.5, 0.5, -0.5, 1],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(target, expected, atol=1e-6))
        self.assertTrue(torch.allclose(target, target.t()))

    def test_ideal_circle_embeddings_have_zero_loss(self):
        angles = build_iemocap_angles()
        embeddings = torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        )
        labels = torch.arange(6)
        loss = CircularCSELoss()(embeddings, labels)
        self.assertLess(float(loss), 1e-12)

    def test_same_class_margin_and_ordered_pair_mean(self):
        angle = math.acos(0.9)
        embeddings = torch.tensor(
            [[1.0, 0.0], [math.cos(angle), math.sin(angle)]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0])
        strict = CircularCSELoss(same_class_margin=0.0)(
            embeddings, labels
        )
        tolerant = CircularCSELoss(same_class_margin=0.1)(
            embeddings, labels
        )
        self.assertAlmostEqual(float(strict), 0.01, places=6)
        self.assertLess(float(tolerant), 1e-12)
        strict.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_singleton_is_differentiable_zero(self):
        embeddings = torch.tensor(
            [[1.0, 0.0]], requires_grad=True
        )
        loss = CircularCSELoss()(embeddings, torch.tensor([0]))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_invalid_inputs_are_rejected(self):
        criterion = CircularCSELoss()
        with self.assertRaises(ValueError):
            criterion(torch.ones(2, 3, 1), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            criterion(torch.ones(2, 3), torch.tensor([0, 6]))
        with self.assertRaises(ValueError):
            criterion(
                torch.tensor([[float("nan"), 0.0]]),
                torch.tensor([0]),
            )
        with self.assertRaises(ValueError):
            CircularCSELoss(same_class_margin=-0.1)


class SelfDistillationLossTest(unittest.TestCase):
    def _outputs(self):
        return {
            "fusion_logits": torch.randn(2, 3, 6, requires_grad=True),
            "text_logits": torch.randn(2, 3, 6, requires_grad=True),
            "audio_logits": torch.randn(2, 3, 6, requires_grad=True),
            "visual_logits": torch.randn(2, 3, 6, requires_grad=True),
            "embeddings": torch.randn(2, 3, 8, requires_grad=True),
        }

    def test_full_objective_preserves_distillation(self):
        outputs = self._outputs()
        labels = torch.tensor([[0, 1, 2], [3, 4, 0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        losses = compute_sdt_cse_losses(
            outputs,
            labels,
            mask,
            iemocap_class_weights(),
            circular_loss_function=CircularCSELoss(),
            circular_weight=0.1,
        )
        expected = (
            losses["fusion_ce"]
            + losses["unimodal_ce"]
            + losses["distillation"]
            + 0.1 * losses["circular_cse"]
        )
        self.assertTrue(
            torch.allclose(losses["total_loss"], expected)
        )
        losses["total_loss"].backward()
        self.assertIsNotNone(outputs["fusion_logits"].grad)
        self.assertIsNotNone(outputs["text_logits"].grad)
        self.assertIsNotNone(outputs["embeddings"].grad)

    def test_padding_does_not_change_kl(self):
        student = torch.randn(1, 3, 6)
        teacher = torch.randn(1, 3, 6)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        first = masked_self_distillation_kl(
            student, teacher, mask, temperature=2.0
        )
        student[:, 2] = 1000.0
        teacher[:, 2] = -1000.0
        second = masked_self_distillation_kl(
            student, teacher, mask, temperature=2.0
        )
        self.assertTrue(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()

