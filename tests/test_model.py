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
    circular_pair_distances,
    iemocap_class_weights,
    minimum_confusion_gap_regularization,
)
from model import (  # noqa: E402
    CIRCLE_ORDER,
    CosineEmotionClassifier,
    LearnableCircularAngles,
    SDTCSEModel,
    SphericalResidualUpdate,
    SphericalFusionHead,
    TransformerEncoder,
)
from train import (  # noqa: E402
    angle_history_row,
    build_argument_parser,
    build_optimizer,
    save_checkpoint,
    spherical_residual_history_row,
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


def make_model(mode, initial_class_angles=None, **kwargs):
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
        **kwargs
    )


class ModelModeTest(unittest.TestCase):
    def test_output_shapes_and_unit_embeddings(self):
        inputs = make_inputs()
        for mode in (
            "sdt",
            "sdt_cosine",
            "sdt_cse",
            "sdt_cse_all_cosine",
            "sdt_cse_all_modal_cse",
            "sdt_cse_fusion_only",
            "sdt_cse_learnable_angles",
            "sdt_cse_learnable_angles_confusion_gap",
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
            if mode in (
                "sdt_cse_all_cosine",
                "sdt_cse_all_modal_cse",
            ):
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
                self.assertEqual(
                    outputs["unimodal_circular_cse_enabled"],
                    mode == "sdt_cse_all_modal_cse",
                )
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
            if mode in (
                "sdt_cse_learnable_angles",
                "sdt_cse_learnable_angles_confusion_gap",
            ):
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
        self.assertTrue(
            torch.allclose(
                losses["total_circular_cse"],
                losses["circular_cse"],
            )
        )
        for name in (
            "text_circular_cse",
            "audio_circular_cse",
            "visual_circular_cse",
            "unimodal_circular_cse",
        ):
            self.assertEqual(float(losses[name]), 0.0)
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

    def test_all_modal_cse_applies_circular_loss_to_four_embeddings(self):
        model = make_model("sdt_cse_all_modal_cse")
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
        expected_circular = (
            losses["fusion_circular_cse"]
            + losses["text_circular_cse"]
            + losses["audio_circular_cse"]
            + losses["visual_circular_cse"]
        )
        expected_total = (
            losses["fusion_ce"]
            + losses["unimodal_ce"]
            + losses["distillation"]
            + 0.1 * expected_circular
        )
        self.assertTrue(
            torch.allclose(
                losses["total_circular_cse"],
                expected_circular,
            )
        )
        self.assertTrue(
            torch.allclose(losses["total_loss"], expected_total)
        )
        for name in (
            "fusion_circular_cse",
            "text_circular_cse",
            "audio_circular_cse",
            "visual_circular_cse",
        ):
            self.assertGreater(float(losses[name]), 0.0)

        losses["total_circular_cse"].backward()
        for projector in (
            model.fusion_projector,
            model.text_projector,
            model.audio_projector,
            model.visual_projector,
        ):
            self.assertIsNotNone(projector.linear_1.weight.grad)
            self.assertGreater(
                float(projector.linear_1.weight.grad.abs().sum()),
                0.0,
            )

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


class SphericalResidualUpdateTest(unittest.TestCase):
    def test_initial_alpha_unit_norm_padding_and_scale_invariance(self):
        torch.manual_seed(19)
        module = SphericalResidualUpdate(
            hidden_dim=4,
            initial_alpha=0.1,
        )
        current = torch.randn(2, 3, 4)
        proposal = torch.randn(2, 3, 4)
        padding_mask = torch.tensor(
            [[False, False, True], [False, True, True]]
        )
        output = module(current, proposal, padding_mask)
        scaled = module(
            current * 7.0,
            proposal * 0.25,
            padding_mask,
        )
        self.assertAlmostEqual(
            float(module.effective_alpha), 0.1, places=6
        )
        self.assertTrue(torch.allclose(output, scaled, atol=1e-6))
        valid_norms = output[~padding_mask].norm(dim=-1)
        self.assertTrue(
            torch.allclose(
                valid_norms,
                torch.ones_like(valid_norms),
                atol=1e-6,
            )
        )
        self.assertEqual(float(output[padding_mask].abs().sum()), 0.0)

    def test_antipodal_fallback_and_zero_proposal_are_finite(self):
        module = SphericalResidualUpdate(
            hidden_dim=3,
            initial_alpha=0.5,
        )
        current = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        antipodal = -current
        output = module(current, antipodal)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output, current, atol=1e-6))
        zero_output = module(current, torch.zeros_like(current))
        self.assertTrue(torch.isfinite(zero_output).all())
        self.assertTrue(
            torch.allclose(
                zero_output.norm(dim=-1),
                torch.ones((1, 2)),
                atol=1e-6,
            )
        )

    def test_gradients_reach_gate_current_and_proposal(self):
        torch.manual_seed(23)
        module = SphericalResidualUpdate(5, initial_alpha=0.2)
        current = torch.randn(2, 3, 5, requires_grad=True)
        proposal = torch.randn(2, 3, 5, requires_grad=True)
        target = torch.randn(2, 3, 5)
        loss = (module(current, proposal) * target).sum()
        loss.backward()
        for gradient in (
            module.alpha_logit.grad,
            current.grad,
            proposal.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            SphericalResidualUpdate(0)
        with self.assertRaises(ValueError):
            SphericalResidualUpdate(4, initial_alpha=0.0)
        with self.assertRaises(ValueError):
            SphericalResidualUpdate(4, eps=0.0)
        module = SphericalResidualUpdate(4)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 4), torch.randn(2, 4))
        with self.assertRaises(ValueError):
            module(torch.randn(2, 3, 4), torch.randn(2, 2, 4))
        with self.assertRaises(ValueError):
            module(
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                torch.zeros(2, 2, dtype=torch.bool),
            )
        current = torch.randn(2, 3, 4)
        current[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            module(current, torch.randn(2, 3, 4))

    def test_model_has_eighteen_independent_gates_and_unit_branches(self):
        model = make_model(
            "sdt_cse_learnable_angles",
            sdt_residual_update="spherical",
            spherical_attention_alpha_init=0.1,
            spherical_mlp_alpha_init=0.2,
        ).eval()
        updates = model.named_spherical_residual_updates()
        self.assertEqual(len(updates), 18)
        self.assertEqual(
            len({id(module.alpha_logit) for _, module in updates}),
            18,
        )
        self.assertEqual(
            sum(name.endswith("attention_update") for name, _ in updates),
            9,
        )
        self.assertEqual(
            sum(name.endswith("mlp_update") for name, _ in updates),
            9,
        )
        for name, module in updates:
            expected = 0.1 if name.endswith("attention_update") else 0.2
            self.assertAlmostEqual(
                float(module.effective_alpha), expected, places=6
            )

        captured = {}
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, TransformerEncoder):
                handles.append(
                    module.register_forward_hook(
                        lambda _module, _inputs, output, key=name: (
                            captured.__setitem__(key, output.detach())
                        )
                    )
                )
        inputs = make_inputs()
        with torch.no_grad():
            model(*inputs)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(captured), 9)
        valid = inputs[3].bool()
        for output in captured.values():
            valid_norms = output[valid].norm(dim=-1)
            self.assertTrue(
                torch.allclose(
                    valid_norms,
                    torch.ones_like(valid_norms),
                    atol=1e-5,
                )
            )
            self.assertEqual(float(output[~valid].abs().sum()), 0.0)

    def test_standard_model_has_no_spherical_parameters(self):
        model = make_model("sdt")
        self.assertEqual(model.named_spherical_residual_updates(), [])
        self.assertFalse(
            any(
                "alpha_logit" in name
                for name, _ in model.named_parameters()
            )
        )

    def test_full_loss_gradients_reach_all_spherical_gates(self):
        model = make_model(
            "sdt_cse_learnable_angles",
            initial_class_angles=build_iemocap_angles(
                geometry="equal"
            ),
            sdt_residual_update="spherical",
        )
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
            angle_weight=0.1,
        )
        losses["total_loss"].backward()
        for _, module in model.named_spherical_residual_updates():
            self.assertIsNotNone(module.alpha_logit.grad)
            self.assertTrue(
                torch.isfinite(module.alpha_logit.grad).all()
            )
            self.assertGreater(
                float(module.alpha_logit.grad.abs().sum()), 0.0
            )

    def test_ce_kl_and_circular_losses_each_reach_spherical_gates(self):
        inputs = make_inputs()
        labels = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 0]])
        for loss_name in ("fusion_ce", "distillation", "circular_cse"):
            model = make_model(
                "sdt_cse",
                sdt_residual_update="spherical",
            )
            outputs = model(*inputs)
            losses = compute_sdt_cse_losses(
                outputs,
                labels,
                inputs[3],
                iemocap_class_weights(),
                circular_loss_function=CircularCSELoss(),
                circular_weight=0.1,
            )
            losses[loss_name].backward()
            for _, module in model.named_spherical_residual_updates():
                gradient = module.alpha_logit.grad
                self.assertIsNotNone(
                    gradient,
                    msg="{} did not reach a gate".format(loss_name),
                )
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(
                    float(gradient.abs().sum()),
                    0.0,
                    msg="{} produced a zero gate gradient".format(
                        loss_name
                    ),
                )

    def test_optimizer_excludes_all_spherical_gates_from_decay(self):
        model = make_model(
            "sdt_cse_learnable_angles",
            sdt_residual_update="spherical",
        )
        optimizer = build_optimizer(model, 1e-4, 1e-5)
        zero_decay_ids = {
            id(parameter)
            for group in optimizer.param_groups
            if group["weight_decay"] == 0.0
            for parameter in group["params"]
        }
        expected_ids = {
            id(parameter)
            for parameter in model.spherical_residual_parameters()
        }
        expected_ids.add(id(model.circular_angle_learner.raw_gaps))
        self.assertTrue(expected_ids.issubset(zero_decay_ids))

    def test_diagnostics_report_all_updates(self):
        model = make_model(
            "sdt_cse",
            sdt_residual_update="spherical",
        ).eval()
        model.enable_spherical_residual_diagnostics(True, reset=True)
        with torch.no_grad():
            model(*make_inputs())
        diagnostics = model.spherical_residual_diagnostics()
        model.enable_spherical_residual_diagnostics(False, reset=False)
        self.assertEqual(len(diagnostics), 18)
        for values in diagnostics.values():
            self.assertEqual(values["valid_state_count"], 6)
            self.assertAlmostEqual(
                values["output_norm_mean"], 1.0, places=5
            )
            self.assertIsNotNone(
                values["mean_angular_movement_degrees"]
            )

    def test_checkpoint_and_history_preserve_spherical_gates(self):
        model = make_model(
            "sdt_cse",
            sdt_residual_update="spherical",
            spherical_attention_alpha_init=0.1,
            spherical_mlp_alpha_init=0.2,
        )
        with torch.no_grad():
            model.t_t.transformer_inter[
                0
            ].attention_update.alpha_logit.add_(0.3)
        optimizer = build_optimizer(model, 1e-4, 1e-5)
        args = build_argument_parser().parse_args(
            [
                "--experiment-mode",
                "sdt_cse",
                "--sdt-residual-update",
                "spherical",
                "--spherical-attention-alpha-init",
                "0.1",
                "--spherical-mlp-alpha-init",
                "0.2",
            ]
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
                epoch=3,
                validation_metrics={"weighted_f1": 60.0},
            )
            checkpoint = torch.load(path, map_location="cpu")
        self.assertEqual(
            checkpoint["sdt_residual_update"], "spherical"
        )
        self.assertEqual(
            len(checkpoint["spherical_residual_gates"]), 18
        )
        restored = make_model(
            "sdt_cse",
            sdt_residual_update="spherical",
            spherical_attention_alpha_init=0.1,
            spherical_mlp_alpha_init=0.2,
        )
        restored.load_state_dict(checkpoint["model_state_dict"])
        for name, alpha in model.spherical_residual_gate_state().items():
            self.assertTrue(
                torch.allclose(
                    alpha,
                    restored.spherical_residual_gate_state()[name],
                )
            )
        history = spherical_residual_history_row(3, restored)
        self.assertEqual(history["epoch"], 3)
        self.assertEqual(len(history), 19)


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

    def test_confusion_gap_mode_adds_only_requested_penalty(self):
        prior = build_iemocap_angles(geometry="equal")
        model = make_model(
            "sdt_cse_learnable_angles_confusion_gap",
            initial_class_angles=prior,
        )
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
            "angle_weight": 0.1,
            "minimum_confusion_gap_degrees": 75.0,
        }
        without_gap = compute_sdt_cse_losses(
            confusion_gap_weight=0.0,
            **common
        )
        with_gap = compute_sdt_cse_losses(
            confusion_gap_weight=0.1,
            **common
        )
        expected_penalty = 2.0 * math.radians(15.0) ** 2
        self.assertAlmostEqual(
            float(with_gap["confusion_gap_regularization"]),
            expected_penalty,
            places=5,
        )
        self.assertTrue(
            torch.allclose(
                with_gap["total_loss"],
                without_gap["total_loss"]
                + 0.1
                * with_gap["confusion_gap_regularization"],
            )
        )
        with_gap["confusion_gap_regularization"].backward()
        gradient = model.circular_angle_learner.raw_gaps.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_confusion_gap_gradient_increases_selected_gaps(self):
        learner = LearnableCircularAngles(
            prior_angles=build_iemocap_angles(geometry="equal")
        )
        optimizer = torch.optim.SGD(learner.parameters(), lr=0.1)
        before = circular_pair_distances(learner()).detach()
        penalty = minimum_confusion_gap_regularization(
            learner(),
            minimum_gap_degrees=75.0,
        )
        penalty.backward()
        optimizer.step()
        after = circular_pair_distances(learner()).detach()
        self.assertTrue(torch.all(after > before))
        self.assertTrue(torch.all(learner.normalized_gaps() > 0))

    def test_confusion_gap_history_records_pair_angles(self):
        prior = build_iemocap_angles(geometry="equal")
        model = make_model(
            "sdt_cse_learnable_angles_confusion_gap",
            initial_class_angles=prior,
        )
        history = angle_history_row(
            1,
            model,
            prior,
            minimum_confusion_gap_degrees=75.0,
        )
        self.assertAlmostEqual(
            history["confusion_gap_happy_excited_degrees"],
            60.0,
            places=4,
        )
        self.assertAlmostEqual(
            history["confusion_gap_angry_frustrated_degrees"],
            60.0,
            places=4,
        )
        self.assertGreater(
            history["confusion_gap_regularization"], 0.0
        )

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

    def test_confusion_gap_checkpoint_contains_constraint_metadata(self):
        prior = build_iemocap_angles(geometry="equal")
        model = make_model(
            "sdt_cse_learnable_angles_confusion_gap",
            initial_class_angles=prior,
        )
        optimizer = build_optimizer(model, 1e-4, 1e-5)
        args = build_argument_parser().parse_args(
            [
                "--experiment-mode",
                "sdt_cse_learnable_angles_confusion_gap",
                "--confusion-gap-weight",
                "1.0",
                "--minimum-confusion-gap-degrees",
                "75",
            ]
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
                epoch=1,
                validation_metrics={"weighted_f1": 1.0},
            )
            checkpoint = torch.load(path, map_location="cpu")
        self.assertEqual(checkpoint["circular_geometry"], "equal")
        self.assertEqual(checkpoint["confusion_gap_weight"], 1.0)
        self.assertEqual(
            checkpoint["minimum_confusion_gap_degrees"], 75.0
        )
        for value in checkpoint[
            "confusion_pair_gaps_degrees"
        ].values():
            self.assertAlmostEqual(value, 60.0, places=4)
        self.assertEqual(len(checkpoint["confusion_pairs"]), 2)


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
