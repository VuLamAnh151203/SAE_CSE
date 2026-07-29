import importlib.util
import math
import os
import sys
import tempfile
import unittest

import torch


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
REPOSITORY_DIR = os.path.dirname(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from losses import (  # noqa: E402
    CircularCSELoss,
    build_iemocap_angles,
    build_target_similarity,
    compute_sdt_cse_losses,
    iemocap_class_weights,
)
from model import (  # noqa: E402
    CIRCLE_ORDER,
    CosineEmotionClassifier,
    LearnableCircularAngles,
    SDTCSEModel,
    SphericalFusionHead,
)
from train import (  # noqa: E402
    angle_history_row,
    build_argument_parser,
    build_optimizer,
    save_checkpoint,
    validate_arguments,
)


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


def make_model(mode, initial_class_angles=None):
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
        initial_class_angles=initial_class_angles,
    )


class ModelModeTest(unittest.TestCase):
    def test_output_shapes_and_unit_embeddings(self):
        inputs = make_inputs()
        for mode in (
            "sdt",
            "sdt_cosine",
            "sdt_cse",
            "sdt_cse_all_cosine",
            "sdt_cse_fusion_only",
            "sdt_cse_learnable_angles",
        ):
            model = make_model(mode).eval()
            with torch.no_grad():
                outputs = model(*inputs)
            self.assertEqual(outputs["fusion_features"].shape, (2, 4, 8))
            self.assertEqual(outputs["fusion_logits"].shape, (2, 4, 6))
            if mode == "sdt_cse_fusion_only":
                self.assertIsNone(outputs["text_logits"])
                self.assertIsNone(outputs["audio_logits"])
                self.assertIsNone(outputs["visual_logits"])
                self.assertIsNone(model.t_output_layer)
                self.assertIsNone(model.a_output_layer)
                self.assertIsNone(model.v_output_layer)
            else:
                self.assertEqual(
                    outputs["text_logits"].shape, (2, 4, 6)
                )
                self.assertEqual(
                    outputs["audio_logits"].shape, (2, 4, 6)
                )
                self.assertEqual(
                    outputs["visual_logits"].shape, (2, 4, 6)
                )
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
            if mode == "sdt_cse_all_cosine":
                self.assertIsInstance(
                    model.t_output_layer, CosineEmotionClassifier
                )
                self.assertIsInstance(
                    model.a_output_layer, CosineEmotionClassifier
                )
                self.assertIsInstance(
                    model.v_output_layer, CosineEmotionClassifier
                )
                for projector in (
                    model.text_projector,
                    model.audio_projector,
                    model.visual_projector,
                ):
                    self.assertIsInstance(projector, SphericalFusionHead)
                for output_name in (
                    "text_embeddings",
                    "audio_embeddings",
                    "visual_embeddings",
                ):
                    modality_embeddings = outputs[output_name]
                    self.assertEqual(
                        modality_embeddings.shape, (2, 4, 6)
                    )
                    modality_norms = modality_embeddings.norm(
                        p=2, dim=-1
                    )
                    self.assertTrue(
                        torch.allclose(
                            modality_norms,
                            torch.ones_like(modality_norms),
                            atol=1e-6,
                        )
                    )
                for scale in (
                    model.effective_unimodal_cosine_scales.values()
                ):
                    self.assertGreaterEqual(float(scale), 1.0)
                    self.assertLessEqual(float(scale), 100.0)
            else:
                self.assertTrue(
                    all(
                        scale is None
                        for scale in (
                            model.effective_unimodal_cosine_scales.values()
                        )
                    )
                )
                self.assertIsNone(outputs["text_embeddings"])
                self.assertIsNone(outputs["audio_embeddings"])
                self.assertIsNone(outputs["visual_embeddings"])
                self.assertIsNone(model.text_projector)
                self.assertIsNone(model.audio_projector)
                self.assertIsNone(model.visual_projector)
            if mode == "sdt_cse_learnable_angles":
                self.assertIsNotNone(model.circular_angle_learner)
                self.assertEqual(outputs["class_angles"].shape, (6,))
                self.assertEqual(outputs["angle_gaps"].shape, (6,))
                self.assertEqual(outputs["angle_offsets"].shape, (6,))
                self.assertEqual(
                    outputs["angle_regularization"].ndim, 0
                )
            else:
                self.assertIsNone(model.circular_angle_learner)
                self.assertIsNone(outputs["class_angles"])
                self.assertIsNone(outputs["angle_gaps"])
                self.assertIsNone(outputs["angle_offsets"])
                self.assertIsNone(outputs["angle_regularization"])

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

    def test_all_cosine_mode_preserves_cse_and_distillation_gradients(self):
        model = make_model("sdt_cse_all_cosine")
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

        self.assertGreater(float(losses["circular_cse"]), 0.0)
        self.assertGreaterEqual(float(losses["distillation"]), 0.0)
        self.assertIsNotNone(model.textf_input.weight.grad)
        self.assertIsNotNone(model.last_gate.fc.weight.grad)
        self.assertIsNotNone(model.fusion_projector.linear_1.weight.grad)
        self.assertIsNotNone(model.cosine_classifier.class_weights.grad)
        self.assertIsNotNone(model.cosine_classifier.log_scale.grad)
        for projector, classifier in (
            (model.text_projector, model.t_output_layer),
            (model.audio_projector, model.a_output_layer),
            (model.visual_projector, model.v_output_layer),
        ):
            self.assertIsNotNone(projector.linear_1.weight.grad)
            self.assertIsNotNone(projector.linear_2.weight.grad)
            self.assertIsNotNone(classifier.class_weights.grad)
            self.assertIsNotNone(classifier.log_scale.grad)

    def test_fusion_only_mode_has_no_unimodal_losses_or_parameters(self):
        model = make_model("sdt_cse_fusion_only")
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
        expected = (
            losses["fusion_ce"] + 0.1 * losses["circular_cse"]
        )
        self.assertTrue(torch.allclose(losses["total_loss"], expected))
        for name in (
            "text_ce",
            "audio_ce",
            "visual_ce",
            "unimodal_ce",
            "text_kl",
            "audio_kl",
            "visual_kl",
            "distillation",
        ):
            self.assertEqual(float(losses[name]), 0.0)
        parameter_names = {
            name for name, _ in model.named_parameters()
        }
        self.assertFalse(
            any("output_layer" in name for name in parameter_names)
        )

        losses["total_loss"].backward()
        self.assertIsNotNone(model.textf_input.weight.grad)
        self.assertIsNotNone(model.last_gate.fc.weight.grad)
        self.assertIsNotNone(model.fusion_projector.linear_1.weight.grad)
        self.assertIsNotNone(model.cosine_classifier.class_weights.grad)

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


class LearnableCircularAnglesTest(unittest.TestCase):
    def test_equal_and_nrc_priors_are_reproduced_at_initialization(self):
        for geometry in ("equal", "nrc_vad"):
            prior = build_iemocap_angles(geometry=geometry)
            learner = LearnableCircularAngles(
                prior_angles=prior,
                circle_order=CIRCLE_ORDER,
            )
            angles = learner()
            gaps = learner.normalized_gaps()
            self.assertTrue(
                torch.allclose(
                    build_target_similarity(angles),
                    build_target_similarity(prior),
                    atol=1e-6,
                )
            )
            self.assertAlmostEqual(float(angles[0]), 0.0, places=6)
            self.assertTrue(torch.all(gaps > 0))
            self.assertAlmostEqual(
                float(gaps.sum()), 2.0 * math.pi, places=5
            )
            ordered = angles[torch.tensor(CIRCLE_ORDER)]
            self.assertTrue(torch.all(ordered[1:] > ordered[:-1]))
            self.assertLess(float(ordered[-1]), 2.0 * math.pi)
            self.assertLess(float(learner.regularization()), 1e-10)

    def test_arbitrary_parameters_preserve_order_and_circumference(self):
        learner = LearnableCircularAngles()
        with torch.no_grad():
            learner.raw_gaps.copy_(
                torch.tensor([-8.0, 4.0, -2.0, 1.0, 7.0, -5.0])
            )
        angles = learner()
        gaps = learner.normalized_gaps()
        ordered = angles[torch.tensor(CIRCLE_ORDER)]
        self.assertTrue(torch.all(gaps > 0))
        self.assertTrue(torch.all(ordered[1:] > ordered[:-1]))
        self.assertAlmostEqual(
            float(gaps.sum()), 2.0 * math.pi, places=5
        )
        self.assertEqual(float(angles[0]), 0.0)

    def test_angle_regularization_has_finite_gap_gradients(self):
        learner = LearnableCircularAngles(
            prior_angles=build_iemocap_angles(geometry="nrc_vad")
        )
        with torch.no_grad():
            learner.raw_gaps[1].add_(0.25)
        regularization = learner.regularization()
        self.assertGreater(float(regularization), 0.0)
        regularization.backward()
        self.assertIsNotNone(learner.raw_gaps.grad)
        self.assertTrue(torch.isfinite(learner.raw_gaps.grad).all())
        self.assertGreater(
            float(learner.raw_gaps.grad.abs().sum()), 0.0
        )

    def test_full_model_preserves_sdt_losses_and_trains_angles(self):
        prior = build_iemocap_angles(geometry="nrc_vad")
        model = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        inputs = make_inputs()
        outputs = model(*inputs)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        losses = compute_sdt_cse_losses(
            outputs,
            labels,
            inputs[3],
            iemocap_class_weights(),
            circular_loss_function=CircularCSELoss(
                class_angles=prior
            ),
            circular_weight=0.1,
            angle_weight=0.1,
        )
        expected = (
            losses["fusion_ce"]
            + losses["unimodal_ce"]
            + losses["distillation"]
            + 0.1 * losses["circular_cse"]
            + 0.1 * losses["angle_regularization"]
        )
        self.assertTrue(torch.allclose(losses["total_loss"], expected))
        self.assertGreater(float(losses["unimodal_ce"]), 0.0)
        self.assertGreaterEqual(float(losses["distillation"]), 0.0)
        losses["total_loss"].backward()
        self.assertIsNotNone(
            model.circular_angle_learner.raw_gaps.grad
        )
        self.assertGreater(
            float(
                model.circular_angle_learner.raw_gaps.grad.abs().sum()
            ),
            0.0,
        )
        self.assertIsNotNone(model.t_output_layer[-1].weight.grad)
        self.assertIsNotNone(model.fusion_projector.linear_1.weight.grad)

    def test_angle_weight_only_adds_prior_penalty(self):
        prior = build_iemocap_angles(geometry="nrc_vad")
        model = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        with torch.no_grad():
            model.circular_angle_learner.raw_gaps[2].add_(0.3)
        inputs = make_inputs()
        outputs = model(*inputs)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        common = {
            "outputs": outputs,
            "labels": labels,
            "utterance_mask": inputs[3],
            "class_weights": iemocap_class_weights(),
            "circular_loss_function": CircularCSELoss(
                class_angles=prior
            ),
            "circular_weight": 0.1,
        }
        without_prior = compute_sdt_cse_losses(
            angle_weight=0.0, **common
        )
        with_prior = compute_sdt_cse_losses(
            angle_weight=0.1, **common
        )
        self.assertTrue(
            torch.allclose(
                with_prior["total_loss"],
                without_prior["total_loss"]
                + 0.1 * with_prior["angle_regularization"],
            )
        )

    def test_zero_circular_and_angle_weights_disable_gap_learning(self):
        prior = build_iemocap_angles(geometry="nrc_vad")
        model = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        inputs = make_inputs()
        outputs = model(*inputs)
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        losses = compute_sdt_cse_losses(
            outputs,
            labels,
            inputs[3],
            iemocap_class_weights(),
            circular_loss_function=CircularCSELoss(
                class_angles=prior
            ),
            circular_weight=0.0,
            angle_weight=0.0,
        )
        losses["total_loss"].backward()
        gap_gradient = model.circular_angle_learner.raw_gaps.grad
        self.assertIsNotNone(gap_gradient)
        self.assertTrue(
            torch.allclose(gap_gradient, torch.zeros_like(gap_gradient))
        )

    def test_state_dict_restores_learned_geometry(self):
        prior = build_iemocap_angles(geometry="nrc_vad")
        original = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        with torch.no_grad():
            original.circular_angle_learner.raw_gaps.add_(
                torch.tensor([0.1, -0.2, 0.3, 0.0, -0.1, 0.2])
            )
        restored = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        restored.load_state_dict(original.state_dict())
        original_state = original.current_circular_angle_state()
        restored_state = restored.current_circular_angle_state()
        for key in ("angles", "gaps", "offsets", "regularization"):
            self.assertTrue(
                torch.allclose(
                    original_state[key], restored_state[key]
                )
            )

    def test_angle_optimizer_group_has_no_weight_decay(self):
        model = make_model("sdt_cse_learnable_angles")
        optimizer = build_optimizer(
            model,
            learning_rate=1e-4,
            weight_decay=1e-5,
        )
        raw_gap_id = id(model.circular_angle_learner.raw_gaps)
        matching_groups = [
            group
            for group in optimizer.param_groups
            if any(
                id(parameter) == raw_gap_id
                for parameter in group["params"]
            )
        ]
        self.assertEqual(len(matching_groups), 1)
        self.assertEqual(matching_groups[0]["weight_decay"], 0.0)
        base_groups = [
            group
            for group in optimizer.param_groups
            if group is not matching_groups[0]
        ]
        self.assertTrue(
            any(group["weight_decay"] == 1e-5 for group in base_groups)
        )

    def test_checkpoint_contains_and_restores_angle_metadata(self):
        prior = build_iemocap_angles(geometry="nrc_vad")
        model = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        with torch.no_grad():
            model.circular_angle_learner.raw_gaps[3].add_(0.2)
        optimizer = build_optimizer(model, 1e-4, 1e-5)
        args = build_argument_parser().parse_args(
            ["--experiment-mode", "sdt_cse_learnable_angles"]
        )
        validate_arguments(args)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            save_checkpoint(
                path,
                model,
                optimizer,
                args,
                {
                    "training": ["train"],
                    "validation": ["valid"],
                    "testing": ["test"],
                },
                epoch=4,
                validation_metrics={"weighted_f1": 70.0},
            )
            checkpoint = torch.load(path, map_location="cpu")

        self.assertEqual(
            checkpoint["circle_order"], list(CIRCLE_ORDER)
        )
        self.assertEqual(checkpoint["angle_weight"], 0.1)
        self.assertEqual(
            checkpoint["selection_protocol"], "validation"
        )
        self.assertEqual(checkpoint["selection_split"], "validation")
        self.assertEqual(
            checkpoint["selection_metrics"]["weighted_f1"], 70.0
        )
        self.assertEqual(len(checkpoint["raw_gaps"]), 6)
        self.assertEqual(len(checkpoint["angle_gaps"]), 6)
        self.assertEqual(len(checkpoint["class_angles"]), 6)
        expected_angles = (
            model.current_circular_angle_state()["angles"]
            .detach()
            .cpu()
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(checkpoint["class_angles"]),
                expected_angles,
            )
        )
        restored = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=prior,
        )
        restored.load_state_dict(checkpoint["model_state_dict"])
        self.assertTrue(
            torch.allclose(
                restored.current_circular_angle_state()["angles"],
                expected_angles,
            )
        )
        history = angle_history_row(4, restored, prior)
        self.assertEqual(history["epoch"], 4)
        self.assertIn("angle_happy_degrees", history)
        self.assertIn(
            "gap_happy_to_excited_degrees", history
        )


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
