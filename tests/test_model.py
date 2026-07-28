import importlib.util
import os
import sys
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
REPOSITORY_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from losses import (  # noqa: E402
    CircularCSELoss,
    compute_sdt_cse_losses,
    iemocap_class_weights,
)
from model import SDTCSEModel  # noqa: E402


def make_inputs():
    torch.manual_seed(7)
    length = 4
    batch = 2
    text = torch.randn(length, batch, 5)
    visual = torch.randn(length, batch, 4)
    audio = torch.randn(length, batch, 3)
    utterance_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    )
    speaker_mask = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    return (
        text,
        visual,
        audio,
        utterance_mask,
        speaker_mask,
        [4, 2],
    )


def make_model(mode):
    return SDTCSEModel(
        d_text=5,
        d_visual=4,
        d_audio=3,
        n_head=2,
        n_classes=6,
        hidden_dim=8,
        n_speakers=2,
        dropout=0.0,
        experiment_mode=mode,
        embedding_dim=6,
        projection_dropout=0.0,
    )


class ModelModeTest(unittest.TestCase):
    def test_output_shapes_and_unit_embeddings(self):
        inputs = make_inputs()
        for mode in ("sdt", "sdt_cosine", "sdt_cse"):
            model = make_model(mode).eval()
            with torch.no_grad():
                outputs = model(*inputs)
            self.assertEqual(outputs["fusion_features"].shape, (2, 4, 8))
            self.assertEqual(outputs["fusion_logits"].shape, (2, 4, 6))
            self.assertEqual(outputs["text_logits"].shape, (2, 4, 6))
            if mode == "sdt":
                self.assertIsNone(outputs["embeddings"])
                self.assertIsNone(model.fusion_projector)
                self.assertIsNone(model.cosine_classifier)
            else:
                self.assertEqual(outputs["embeddings"].shape, (2, 4, 6))
                norms = outputs["embeddings"].norm(p=2, dim=-1)
                self.assertTrue(
                    torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
                )
                self.assertGreaterEqual(
                    float(model.effective_cosine_scale), 1.0
                )
                self.assertLessEqual(
                    float(model.effective_cosine_scale), 100.0
                )

    def test_cse_gradients_reach_encoder_and_unimodal_heads(self):
        model = make_model("sdt_cse")
        inputs = make_inputs()
        outputs = model(*inputs)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        losses = compute_sdt_cse_losses(
            outputs,
            labels,
            inputs[3],
            iemocap_class_weights(),
            circular_loss_function=CircularCSELoss(),
            circular_weight=0.1,
        )
        losses["total_loss"].backward()
        self.assertIsNotNone(model.textf_input.weight.grad)
        self.assertIsNotNone(model.last_gate.fc.weight.grad)
        self.assertIsNotNone(model.fusion_projector.linear_1.weight.grad)
        self.assertIsNotNone(model.cosine_classifier.class_weights.grad)
        self.assertIsNotNone(model.t_output_layer[-1].weight.grad)

    def test_cosine_control_has_no_circular_term(self):
        model = make_model("sdt_cosine")
        inputs = make_inputs()
        outputs = model(*inputs)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        losses = compute_sdt_cse_losses(
            outputs,
            labels,
            inputs[3],
            iemocap_class_weights(),
            circular_loss_function=None,
            circular_weight=0.0,
        )
        expected = (
            losses["fusion_ce"]
            + losses["unimodal_ce"]
            + losses["distillation"]
        )
        self.assertTrue(torch.allclose(losses["total_loss"], expected))
        self.assertEqual(float(losses["circular_cse"]), 0.0)


class OriginalSDTParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(REPOSITORY_DIR, "SDT", "model.py")
        specification = importlib.util.spec_from_file_location(
            "original_sdt_model_for_test", model_path
        )
        cls.original_module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.original_module)

    def _paired_models(self):
        torch.manual_seed(11)
        original = self.original_module.Transformer_Based_Model(
            "IEMOCAP",
            1,
            5,
            4,
            3,
            2,
            n_classes=6,
            hidden_dim=8,
            n_speakers=2,
            dropout=0.0,
            appraisal_mode="none",
        ).eval()
        candidate = make_model("sdt").eval()
        candidate_state = candidate.state_dict()
        copied = {}
        for name, value in original.state_dict().items():
            if name in candidate_state and candidate_state[name].shape == value.shape:
                copied[name] = value
        candidate_state.update(copied)
        candidate.load_state_dict(candidate_state)
        return original, candidate

    def test_linear_mode_matches_original_fusion_and_logits(self):
        original, candidate = self._paired_models()

        captured = {}

        def capture_fusion(module, module_inputs, module_output):
            captured["fusion"] = module_output

        handle = original.last_gate.register_forward_hook(capture_fusion)
        inputs = make_inputs()
        with torch.no_grad():
            original_outputs = original(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
            )
            candidate_outputs = candidate(*inputs)
        handle.remove()

        self.assertTrue(
            torch.allclose(
                captured["fusion"],
                candidate_outputs["fusion_features"],
                atol=1e-6,
            )
        )
        original_fusion_log_probability = original_outputs[3]
        candidate_log_probability = torch.log_softmax(
            candidate_outputs["fusion_logits"], dim=-1
        )
        self.assertTrue(
            torch.allclose(
                original_fusion_log_probability,
                candidate_log_probability,
                atol=1e-6,
            )
        )
        for original_index, candidate_name in (
            (0, "text_logits"),
            (1, "audio_logits"),
            (2, "visual_logits"),
        ):
            self.assertTrue(
                torch.allclose(
                    original_outputs[original_index],
                    torch.log_softmax(
                        candidate_outputs[candidate_name], dim=-1
                    ),
                    atol=1e-6,
                )
            )

    def test_linear_mode_matches_original_total_loss(self):
        original, candidate = self._paired_models()
        inputs = make_inputs()
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        with torch.no_grad():
            original_outputs = original(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
            )
            candidate_outputs = candidate(*inputs)

        weights = iemocap_class_weights()
        original_nll = self.original_module.MaskedNLLLoss(weights)
        original_kl = self.original_module.MaskedKLDivLoss()
        flat_labels = labels.reshape(-1)
        original_loss = (
            original_nll(
                original_outputs[3].reshape(-1, 6),
                flat_labels,
                inputs[3],
            )
            + original_nll(
                original_outputs[0].reshape(-1, 6),
                flat_labels,
                inputs[3],
            )
            + original_nll(
                original_outputs[1].reshape(-1, 6),
                flat_labels,
                inputs[3],
            )
            + original_nll(
                original_outputs[2].reshape(-1, 6),
                flat_labels,
                inputs[3],
            )
            + original_kl(
                original_outputs[5].reshape(-1, 6),
                original_outputs[8].reshape(-1, 6),
                inputs[3],
            )
            + original_kl(
                original_outputs[6].reshape(-1, 6),
                original_outputs[8].reshape(-1, 6),
                inputs[3],
            )
            + original_kl(
                original_outputs[7].reshape(-1, 6),
                original_outputs[8].reshape(-1, 6),
                inputs[3],
            )
        )
        candidate_losses = compute_sdt_cse_losses(
            candidate_outputs,
            labels,
            inputs[3],
            weights,
            circular_weight=0.0,
        )
        self.assertTrue(
            torch.allclose(
                original_loss,
                candidate_losses["total_loss"],
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
