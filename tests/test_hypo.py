import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from aggregate_results import condition_name  # noqa: E402
from analyze_geometry import compute_geometry_metrics  # noqa: E402
from losses import (  # noqa: E402
    CircularCSELoss,
    HypoPrototypeLoss,
    build_iemocap_angles,
    compute_sdt_cse_losses,
    iemocap_class_weights,
)
from model import (  # noqa: E402
    CIRCULAR_CSE_MODES,
    HYPO_ALIGNMENT_MODES,
    HYPO_MODES,
    SDTCSEModel,
)
from train import (  # noqa: E402
    build_argument_parser,
    build_bilevel_angle_optimizer,
    bilevel_angle_step,
    experiment_directory_name,
    save_checkpoint,
    save_hypo_artifacts,
    validate_arguments,
)


def six_embeddings(requires_grad=False):
    return torch.tensor(
        [
            [1.0, 0.2, 0.1],
            [0.2, 1.0, 0.3],
            [0.1, 0.3, 1.0],
            [-1.0, 0.1, 0.2],
            [0.2, -1.0, 0.1],
            [0.1, 0.2, -1.0],
        ],
        requires_grad=requires_grad,
    )


def minimal_outputs(embeddings, fusion_logits=None):
    count = embeddings.size(0)
    if fusion_logits is None:
        fusion_logits = torch.zeros(
            1, count, 6, requires_grad=True
        )
    return {
        "fusion_logits": fusion_logits,
        "fusion_cosine_scores": fusion_logits,
        "embeddings": embeddings.reshape(
            1, count, embeddings.size(-1)
        ),
        "text_logits": None,
        "audio_logits": None,
        "visual_logits": None,
    }


class HypoPrototypeLossTest(unittest.TestCase):
    def test_losses_match_direct_formulas(self):
        temperature = 0.4
        embeddings = six_embeddings()
        labels = torch.arange(6)
        criterion = HypoPrototypeLoss(
            6, 3, temperature=temperature, prototype_momentum=0.0
        )
        components = criterion(
            embeddings, labels, update_prototypes=True
        )
        normalized = F.normalize(embeddings, dim=-1)
        expected_compactness = F.cross_entropy(
            normalized @ normalized.t() / temperature,
            labels,
        )
        similarities = normalized @ normalized.t() / temperature
        off_diagonal = ~torch.eye(6, dtype=torch.bool)
        expected_dispersion = (
            torch.logsumexp(
                similarities[off_diagonal].reshape(6, 5),
                dim=1,
            )
            - math.log(5)
        ).mean()
        self.assertTrue(components["active"])
        self.assertTrue(
            torch.allclose(
                components["compactness"],
                expected_compactness,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                components["dispersion"],
                expected_dispersion,
                atol=1e-6,
            )
        )

    def test_initialization_ema_unit_norm_and_order_invariance(self):
        initial = six_embeddings()
        labels = torch.arange(6)
        first = HypoPrototypeLoss(
            6, 3, temperature=0.1, prototype_momentum=0.75
        )
        second = HypoPrototypeLoss(
            6, 3, temperature=0.1, prototype_momentum=0.75
        )
        first(initial, labels, update_prototypes=True)
        second(initial.flip(0), labels.flip(0), update_prototypes=True)
        expected_initial = F.normalize(initial, dim=-1)
        self.assertTrue(
            torch.allclose(first.prototypes, expected_initial)
        )
        self.assertTrue(
            torch.allclose(first.prototypes, second.prototypes)
        )

        update = torch.cat(
            (initial + 0.3, initial * 0.5 - 0.2), dim=0
        )
        update_labels = torch.arange(6).repeat(2)
        class_zero_mean = F.normalize(
            F.normalize(
                update[update_labels == 0], dim=-1
            ).mean(dim=0),
            dim=0,
        )
        expected_zero = F.normalize(
            0.75 * expected_initial[0] + 0.25 * class_zero_mean,
            dim=0,
        )
        first(update, update_labels, update_prototypes=True)
        permutation = torch.tensor(
            [9, 0, 7, 4, 11, 2, 6, 5, 1, 10, 3, 8]
        )
        second(
            update[permutation],
            update_labels[permutation],
            update_prototypes=True,
        )
        self.assertTrue(
            torch.allclose(first.prototypes, second.prototypes)
        )
        self.assertTrue(
            torch.allclose(first.prototypes[0], expected_zero)
        )
        self.assertTrue(
            torch.allclose(
                first.prototypes.norm(dim=-1), torch.ones(6)
            )
        )
        self.assertTrue(
            torch.equal(first.update_counts, torch.full((6,), 2))
        )

    def test_partial_coverage_is_zero_then_full_coverage_activates(self):
        embeddings = six_embeddings(requires_grad=True)
        criterion = HypoPrototypeLoss(6, 3)
        partial = criterion(
            embeddings[:5],
            torch.arange(5),
            update_prototypes=True,
        )
        self.assertFalse(partial["active"])
        self.assertEqual(
            float(partial["compactness"].detach()), 0.0
        )
        self.assertEqual(
            float(partial["dispersion"].detach()), 0.0
        )
        (
            partial["compactness"] + partial["dispersion"]
        ).backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertEqual(float(embeddings.grad.abs().sum()), 0.0)

        complete = criterion(
            embeddings[5:],
            torch.tensor([5]),
            update_prototypes=True,
        )
        self.assertTrue(complete["active"])
        self.assertEqual(criterion.initialized_classes, 6)
        self.assertEqual(criterion.prototype_coverage, 1.0)

    def test_compactness_and_dispersion_have_embedding_gradients(self):
        embeddings = six_embeddings(requires_grad=True)
        criterion = HypoPrototypeLoss(
            6, 3, temperature=0.3, prototype_momentum=0.0
        )
        components = criterion(
            embeddings, torch.arange(6), update_prototypes=True
        )
        compactness_gradient = torch.autograd.grad(
            components["compactness"],
            embeddings,
            retain_graph=True,
        )[0]
        dispersion_gradient = torch.autograd.grad(
            components["dispersion"], embeddings
        )[0]
        for gradient in (
            compactness_gradient,
            dispersion_gradient,
        ):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_padding_is_ignored_and_evaluation_does_not_update(self):
        valid_embeddings = six_embeddings()
        padded = torch.tensor([[100.0, -200.0, 300.0]])
        embeddings = torch.cat((valid_embeddings, padded), dim=0)
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0.0]])
        criterion = HypoPrototypeLoss(6, 3)
        compute_sdt_cse_losses(
            minimal_outputs(embeddings),
            labels,
            mask,
            torch.ones(6),
            hypo_loss_function=criterion,
            hypo_loss_weight=0.1,
            update_hypo_prototypes=True,
        )
        self.assertTrue(
            torch.allclose(
                criterion.prototypes[0],
                F.normalize(valid_embeddings[0], dim=0),
            )
        )
        before = {
            key: value.clone()
            for key, value in criterion.state_dict().items()
        }
        evaluation_embeddings = torch.randn(7, 3)
        compute_sdt_cse_losses(
            minimal_outputs(evaluation_embeddings),
            labels,
            mask,
            torch.ones(6),
            hypo_loss_function=criterion,
            hypo_loss_weight=0.1,
            update_hypo_prototypes=False,
        )
        for key, value in criterion.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_total_loss_composition(self):
        embeddings = six_embeddings(requires_grad=True)
        criterion = HypoPrototypeLoss(
            6, 3, temperature=0.2, prototype_momentum=0.0
        )
        losses = compute_sdt_cse_losses(
            minimal_outputs(embeddings),
            torch.arange(6).reshape(1, 6),
            torch.ones(1, 6),
            torch.ones(6),
            hypo_loss_function=criterion,
            hypo_loss_weight=0.3,
            hypo_compactness_weight=2.5,
            update_hypo_prototypes=True,
        )
        expected_hypo = (
            2.5 * losses["hypo_compactness"]
            + losses["hypo_dispersion"]
        )
        self.assertTrue(
            torch.allclose(losses["hypo_total"], expected_hypo)
        )
        self.assertTrue(
            torch.allclose(
                losses["total_loss"],
                losses["fusion_ce"] + 0.3 * expected_hypo,
            )
        )

    def test_circle_alignment_replaces_unrestricted_dispersion(self):
        criterion = HypoPrototypeLoss(
            6, 3, temperature=0.2, prototype_momentum=0.5
        )
        criterion(
            six_embeddings(),
            torch.arange(6),
            update_prototypes=True,
        )
        before = {
            key: value.clone()
            for key, value in criterion.state_dict().items()
        }
        embeddings = (six_embeddings() + 0.15).requires_grad_()
        outputs = minimal_outputs(embeddings)
        outputs["class_angles"] = build_iemocap_angles()
        losses = compute_sdt_cse_losses(
            outputs,
            torch.arange(6).reshape(1, 6),
            torch.ones(1, 6),
            torch.ones(6),
            hypo_loss_function=criterion,
            hypo_loss_weight=0.3,
            hypo_compactness_weight=2.0,
            use_batch_hypo_candidates=True,
            hypo_alignment_enabled=True,
            hypo_alignment_weight=0.7,
        )
        expected_hypo = (
            2.0 * losses["hypo_compactness"]
            + 0.7 * losses["hypo_alignment"]
        )
        self.assertEqual(
            float(losses["hypo_dispersion"].detach()), 0.0
        )
        self.assertGreater(
            float(losses["hypo_alignment"].detach()), 0.0
        )
        self.assertTrue(
            torch.allclose(losses["hypo_total"], expected_hypo)
        )
        alignment_gradient = torch.autograd.grad(
            losses["hypo_alignment"], embeddings
        )[0]
        self.assertTrue(torch.isfinite(alignment_gradient).all())
        self.assertGreater(
            float(alignment_gradient.abs().sum()), 0.0
        )
        for key, value in criterion.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


class HypoIntegrationTest(unittest.TestCase):
    @staticmethod
    def _model():
        return SDTCSEModel(
            d_text=5,
            d_visual=4,
            d_audio=3,
            n_head=2,
            n_classes=6,
            hidden_dim=8,
            n_speakers=2,
            dropout=0.0,
            experiment_mode="sdt_hypo",
            embedding_dim=3,
            projection_dropout=0.0,
        )

    def test_mode_architecture_is_prior_free(self):
        self.assertIn("sdt_hypo", HYPO_MODES)
        self.assertNotIn("sdt_hypo", CIRCULAR_CSE_MODES)
        model = self._model()
        self.assertIsNotNone(model.fusion_projector)
        self.assertIsNotNone(model.cosine_classifier)
        self.assertIsNone(model.circular_angle_learner)
        self.assertIsNotNone(model.t_output_layer)
        self.assertIsNone(model.text_projector)

    def test_checkpoint_round_trip_restores_bank_and_predictions(self):
        torch.manual_seed(13)
        model = self._model()
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = HypoPrototypeLoss(6, 3)
        reference_components = criterion(
            six_embeddings(),
            torch.arange(6),
            update_prototypes=True,
        )
        text = torch.randn(3, 1, 5)
        visual = torch.randn(3, 1, 4)
        audio = torch.randn(3, 1, 3)
        mask = torch.ones(1, 3)
        speakers = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
        )
        with torch.no_grad():
            reference_logits = model(
                text,
                visual,
                audio,
                mask,
                speakers,
                [3],
            )["fusion_logits"]

        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--experiment-mode",
                "sdt_hypo",
                "--hidden-dim",
                "8",
                "--n-head",
                "2",
                "--embedding-dim",
                "3",
                "--dropout",
                "0",
                "--projection-dropout",
                "0",
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
                {"training": [], "validation": [], "testing": []},
                epoch=2,
                hypo_loss_function=criterion,
            )
            checkpoint = torch.load(path, map_location="cpu")
        self.assertIsNone(checkpoint["circular_geometry"])
        self.assertIsNone(checkpoint["prior_class_angles"])
        self.assertIsNone(checkpoint["class_angles"])
        self.assertIsNone(checkpoint["target_similarity"])

        restored_model = self._model()
        restored_model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        restored_model.eval()
        restored_criterion = HypoPrototypeLoss(6, 3)
        restored_criterion.load_state_dict(
            checkpoint["hypo_state_dict"]
        )
        with torch.no_grad():
            restored_logits = restored_model(
                text,
                visual,
                audio,
                mask,
                speakers,
                [3],
            )["fusion_logits"]
            restored_components = restored_criterion(
                six_embeddings(),
                torch.arange(6),
                update_prototypes=False,
            )
        self.assertTrue(
            torch.equal(
                criterion.update_counts,
                restored_criterion.update_counts,
            )
        )
        self.assertTrue(
            torch.equal(
                criterion.prototypes,
                restored_criterion.prototypes,
            )
        )
        self.assertTrue(
            torch.allclose(reference_logits, restored_logits)
        )
        for name in ("compactness", "dispersion"):
            self.assertTrue(
                torch.allclose(
                    reference_components[name],
                    restored_components[name],
                )
            )

    def test_arguments_directory_and_aggregation_names(self):
        parser = build_argument_parser()
        args = parser.parse_args(
            ["--experiment-mode", "sdt_hypo"]
        )
        validate_arguments(args)
        self.assertEqual(args.circular_weight, 0.0)
        self.assertEqual(args.angle_weight, 0.0)
        self.assertEqual(args.confusion_gap_weight, 0.0)
        self.assertEqual(args.confused_cse_pair_weight, 0.0)
        self.assertEqual(args.confusion_classification_weight, 0.0)
        self.assertEqual(
            experiment_directory_name(
                "sdt_hypo",
                0.0,
                hypo_loss_weight=0.1,
                hypo_compactness_weight=2.0,
                hypo_temperature=0.1,
                hypo_prototype_momentum=0.95,
            ),
            "sdt_hypo_lambda_0.1_w2_tau0.1_pm0.95",
        )
        summary = {
            "experiment_mode": "sdt_hypo",
            "hypo_loss_weight": 0.1,
            "hypo_compactness_weight": 2.0,
            "hypo_temperature": 0.1,
            "hypo_prototype_momentum": 0.95,
        }
        self.assertEqual(
            condition_name(summary),
            "sdt_hypo_lambda_0.1_w2_tau0.1_pm0.95",
        )

        for option, value in (
            ("--hypo-loss-weight", "nan"),
            ("--hypo-compactness-weight", "-1"),
            ("--hypo-temperature", "0"),
            ("--hypo-prototype-momentum", "1"),
        ):
            invalid = parser.parse_args(
                ["--experiment-mode", "sdt_hypo", option, value]
            )
            with self.assertRaises(ValueError):
                validate_arguments(invalid)

    def test_aligned_bilevel_mode_keeps_both_geometries(self):
        mode = "sdt_cse_bilevel_confusion_gap_hypo_aligned"
        self.assertIn(mode, HYPO_MODES)
        self.assertIn(mode, HYPO_ALIGNMENT_MODES)
        self.assertIn(mode, CIRCULAR_CSE_MODES)
        model = SDTCSEModel(
            d_text=5,
            d_visual=4,
            d_audio=3,
            n_head=2,
            n_classes=6,
            hidden_dim=8,
            n_speakers=2,
            dropout=0.0,
            experiment_mode=mode,
            embedding_dim=3,
            projection_dropout=0.0,
            bilevel_gap_minimum_degrees=50.0,
            bilevel_gap_maximum_degrees=150.0,
            bilevel_gap_initial_degrees=90.0,
        )
        self.assertIsNotNone(model.circular_angle_learner)
        self.assertIsNotNone(model.fusion_projector)
        self.assertIsNotNone(model.t_output_layer)

        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--experiment-mode",
                mode,
                "--bilevel-gap-minimum-degrees",
                "50",
                "--bilevel-gap-maximum-degrees",
                "150",
                "--bilevel-gap-initial-degrees",
                "90",
            ]
        )
        validate_arguments(args)
        self.assertEqual(args.circular_weight, 0.1)
        self.assertEqual(args.hypo_loss_weight, 0.1)
        self.assertEqual(args.hypo_alignment_weight, 1.0)
        condition = experiment_directory_name(
            mode,
            args.circular_weight,
            args.circular_geometry,
            bilevel_gap_minimum_degrees=50.0,
            bilevel_gap_maximum_degrees=150.0,
            bilevel_gap_initial_degrees=90.0,
            confused_cse_pair_weight=5.0,
            confusion_classification_margin=0.1,
            confusion_classification_weight=0.1,
            bilevel_angle_learning_rate=0.001,
            bilevel_outer_confusion_weight=0.1,
            hypo_loss_weight=0.1,
            hypo_compactness_weight=2.0,
            hypo_alignment_weight=1.0,
            hypo_temperature=0.1,
            hypo_prototype_momentum=0.95,
        )
        self.assertIn("_range_50-150_init_90_", condition)
        self.assertIn(
            "_hlambda_0.1_w2_a1_tau0.1_pm0.95",
            condition,
        )
        summary = {
            "experiment_mode": mode,
            "circular_weight": 0.1,
            "confused_cse_pair_weight": 5.0,
            "confusion_classification_margin": 0.1,
            "confusion_classification_weight": 0.1,
            "hypo_loss_weight": 0.1,
            "hypo_compactness_weight": 2.0,
            "hypo_alignment_weight": 1.0,
            "hypo_temperature": 0.1,
            "hypo_prototype_momentum": 0.95,
            "bilevel_geometry": {
                "minimum_degrees": 50.0,
                "maximum_degrees": 150.0,
                "initial_degrees": 90.0,
                "angle_learning_rate": 0.001,
                "outer_confusion_weight": 0.1,
            },
        }
        self.assertEqual(condition_name(summary), condition)

    def test_bilevel_hvp_never_commits_hypo_candidates(self):
        torch.manual_seed(29)
        mode = "sdt_cse_bilevel_confusion_gap_hypo_aligned"
        model = SDTCSEModel(
            d_text=5,
            d_visual=4,
            d_audio=3,
            n_head=2,
            n_classes=6,
            hidden_dim=8,
            n_speakers=2,
            dropout=0.0,
            experiment_mode=mode,
            embedding_dim=3,
            projection_dropout=0.0,
            bilevel_gap_minimum_degrees=50.0,
            bilevel_gap_maximum_degrees=150.0,
            bilevel_gap_initial_degrees=90.0,
        )
        utterance_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
        )
        speaker_mask = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ]
        )
        batch = {
            "text": torch.randn(4, 2, 5),
            "visual": torch.randn(4, 2, 4),
            "audio": torch.randn(4, 2, 3),
            "utterance_mask": utterance_mask,
            "speaker_mask": speaker_mask.permute(1, 0, 2),
            "labels": torch.tensor(
                [[0, 4, 3, 5], [1, 2, 0, 0]]
            ),
        }
        validation_batch = {
            key: value.clone() for key, value in batch.items()
        }
        validation_batch["text"] += 0.03
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--experiment-mode",
                mode,
                "--bilevel-gap-minimum-degrees",
                "50",
                "--bilevel-gap-maximum-degrees",
                "150",
                "--bilevel-gap-initial-degrees",
                "90",
                "--bilevel-angle-learning-rate",
                "0.01",
                "--bilevel-inner-step-size",
                "0.001",
                "--bilevel-hvp-radius",
                "0.01",
            ]
        )
        validate_arguments(args)
        hypo = HypoPrototypeLoss(6, 3)
        hypo(
            six_embeddings(),
            torch.arange(6),
            update_prototypes=True,
        )
        counts_before = hypo.update_counts.clone()
        prototypes_before = hypo.prototypes.clone()
        angles = model.current_circular_angle_state()[
            "angles"
        ].detach()
        circular = CircularCSELoss(
            class_angles=angles,
            confusion_pair_weight=5.0,
        )
        metrics = bilevel_angle_step(
            model,
            batch,
            validation_batch,
            iemocap_class_weights(),
            circular,
            args,
            build_bilevel_angle_optimizer(model, 0.01),
            hypo_loss_function=hypo,
        )
        self.assertTrue(
            math.isfinite(metrics["bilevel_hypergradient_norm"])
        )
        self.assertTrue(torch.equal(hypo.update_counts, counts_before))
        self.assertTrue(
            torch.equal(hypo.prototypes, prototypes_before)
        )

    def test_artifacts_and_prototype_target_geometry(self):
        criterion = HypoPrototypeLoss(6, 3)
        embeddings = six_embeddings()
        criterion(
            embeddings,
            torch.arange(6),
            update_prototypes=True,
        )
        parser = build_argument_parser()
        args = parser.parse_args(
            ["--experiment-mode", "sdt_hypo"]
        )
        validate_arguments(args)
        with tempfile.TemporaryDirectory() as directory:
            payload = save_hypo_artifacts(
                directory, criterion, args, selected_epoch=4
            )
            with np.load(
                os.path.join(directory, "hypo_prototypes.npz")
            ) as archive:
                self.assertTrue(
                    np.allclose(
                        np.linalg.norm(
                            archive["prototypes"], axis=1
                        ),
                        1.0,
                    )
                )
            with open(
                os.path.join(directory, "hypo_geometry.json"),
                encoding="utf-8",
            ) as source:
                geometry = json.load(source)
            self.assertEqual(geometry["selected_epoch"], 4)
            self.assertEqual(geometry["update_counts"], [1] * 6)

            metrics, _, target, _ = compute_geometry_metrics(
                embeddings.detach().numpy(),
                np.arange(6),
                target_similarity=payload[
                    "prototype_cosine_similarity"
                ],
            )
            self.assertEqual(target.shape, (6, 6))
            self.assertIn("cross_class_mae", metrics)


if __name__ == "__main__":
    unittest.main()
