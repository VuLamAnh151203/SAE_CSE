import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from analyze_geometry import save_geometry_artifacts
from dataloader import (
    DEFAULT_FEATURE_PATH,
    SELECTION_PROTOCOLS,
    create_iemocap_loaders,
)
from losses import (
    CIRCULAR_GEOMETRIES,
    CircularCSELoss,
    EMOTION_NAMES,
    HypoPrototypeLoss,
    ID2EMOTION,
    IEMOCAP_CONFUSION_PAIRS,
    build_iemocap_angles,
    build_iemocap_vad_anchors,
    build_target_similarity,
    circular_pair_distances,
    compute_sdt_cse_losses,
    iemocap_class_weights,
    masked_confusion_classification_margin,
    masked_weighted_cross_entropy,
    minimum_confusion_gap_regularization,
)
from model import (
    BILEVEL_ALL_GAP_MODES,
    BILEVEL_ANGLE_MODES,
    BILEVEL_SHARED_GAP_MODES,
    BilevelAllGapsAngles,
    CIRCLE_ORDER,
    CIRCULAR_CSE_MODES,
    CONFUSION_GAP_MODES,
    CONFUSION_MARGIN_MODES,
    EXPERIMENT_MODES,
    FUSION_ONLY_MODES,
    HYPO_MODES,
    LEARNABLE_ANGLE_MODES,
    SDT_RESIDUAL_UPDATES,
    SDTCSEModel,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "results")
)
LOSS_NAMES = (
    "total_loss",
    "fusion_ce",
    "text_ce",
    "audio_ce",
    "visual_ce",
    "unimodal_ce",
    "text_kl",
    "audio_kl",
    "visual_kl",
    "distillation",
    "circular_cse",
    "fusion_circular_cse",
    "text_circular_cse",
    "audio_circular_cse",
    "visual_circular_cse",
    "unimodal_circular_cse",
    "total_circular_cse",
    "angle_regularization",
    "gap_prior_regularization",
    "confusion_gap_regularization",
    "confusion_classification_margin",
    "hypo_compactness",
    "hypo_dispersion",
    "hypo_total",
)
CONFUSION_PAIR_NAMES = (
    ("happy_excited", 0, 4),
    ("angry_frustrated", 3, 5),
)
CONFUSION_PAIR_METRIC_NAMES = tuple(
    "{}_{}".format(pair_name, suffix)
    for pair_name, _, _ in CONFUSION_PAIR_NAMES
    for suffix in (
        "pair_macro_f1",
        "pair_weighted_f1",
        "mutual_confusion_rate",
    )
)
BILEVEL_METRIC_NAMES = (
    "bilevel_outer_classification_loss",
    "bilevel_gap_prior_regularization",
    "bilevel_hypergradient",
    "bilevel_hypergradient_norm",
    "bilevel_gap_before_degrees",
    "bilevel_gap_after_degrees",
    "bilevel_gap_change_l2_degrees",
    "bilevel_updates",
)


def uses_validation_gap_learning(args):
    if args.experiment_mode in BILEVEL_SHARED_GAP_MODES:
        return True
    return (
        args.experiment_mode in BILEVEL_ALL_GAP_MODES
        and getattr(
            args, "all_gap_learning_source", "validation"
        )
        == "validation"
    )


def set_random_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name, gpu_id=0):
    if gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative")
    use_cuda = device_name == "cuda" or (
        device_name == "auto" and torch.cuda.is_available()
    )
    if not use_cuda:
        if device_name == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        raise ValueError(
            "--gpu-id {} is invalid; {} CUDA device(s) are visible".format(
                gpu_id, device_count
            )
        )
    torch.cuda.set_device(gpu_id)
    return torch.device("cuda:{}".format(gpu_id))


def move_batch_to_device(batch, device):
    moved = dict(batch)
    for key in (
        "text",
        "visual",
        "audio",
        "speaker_mask",
        "utterance_mask",
        "labels",
    ):
        moved[key] = batch[key].to(device)
    return moved


def cycle_batches(dataloader):
    while True:
        produced = False
        for batch in dataloader:
            produced = True
            yield batch
        if not produced:
            raise RuntimeError(
                "cannot cycle an empty validation dataloader"
            )


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.size == 0:
        return {
            "accuracy": float("nan"),
            "weighted_f1": float("nan"),
            "macro_f1": float("nan"),
            "macro_recall": float("nan"),
        }
    return {
        "accuracy": float(accuracy_score(labels, predictions) * 100.0),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
            * 100.0
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
            * 100.0
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
            * 100.0
        ),
    }


def confusion_pair_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    result = {}
    for pair_name, first_id, second_id in CONFUSION_PAIR_NAMES:
        selected = np.isin(labels, (first_id, second_id))
        pair_labels = labels[selected]
        pair_predictions = predictions[selected]
        prefix = "{}_".format(pair_name)
        if pair_labels.size == 0:
            result[prefix + "pair_macro_f1"] = float("nan")
            result[prefix + "pair_weighted_f1"] = float("nan")
            result[prefix + "mutual_confusion_rate"] = float("nan")
            continue
        result[prefix + "pair_macro_f1"] = float(
            f1_score(
                pair_labels,
                pair_predictions,
                labels=[first_id, second_id],
                average="macro",
                zero_division=0,
            )
            * 100.0
        )
        result[prefix + "pair_weighted_f1"] = float(
            f1_score(
                pair_labels,
                pair_predictions,
                labels=[first_id, second_id],
                average="weighted",
                zero_division=0,
            )
            * 100.0
        )
        mutual_confusions = (
            (pair_labels == first_id)
            & (pair_predictions == second_id)
        ) | (
            (pair_labels == second_id)
            & (pair_predictions == first_id)
        )
        result[prefix + "mutual_confusion_rate"] = float(
            mutual_confusions.mean() * 100.0
        )
    return result


def _model_forward(model, batch):
    lengths = (
        batch["utterance_mask"].sum(dim=1).long().cpu().tolist()
    )
    return model(
        batch["text"],
        batch["visual"],
        batch["audio"],
        batch["utterance_mask"],
        batch["speaker_mask"].permute(1, 0, 2),
        lengths,
    )


def _compute_batch_losses(
    model,
    batch,
    class_weights,
    circular_loss_function,
    args,
    hypo_loss_function=None,
    update_hypo_prototypes=False,
):
    outputs = _model_forward(model, batch)
    losses = compute_sdt_cse_losses(
        outputs,
        batch["labels"],
        batch["utterance_mask"],
        class_weights,
        circular_loss_function=circular_loss_function,
        temperature=args.temperature,
        fusion_ce_weight=args.fusion_ce_weight,
        unimodal_ce_weight=args.unimodal_ce_weight,
        distillation_weight=args.distillation_weight,
        circular_weight=args.circular_weight,
        angle_weight=args.angle_weight,
        confusion_gap_weight=args.confusion_gap_weight,
        minimum_confusion_gap_degrees=(
            args.minimum_confusion_gap_degrees
        ),
        confusion_classification_weight=(
            args.confusion_classification_weight
        ),
        confusion_classification_margin=(
            args.confusion_classification_margin
        ),
        hypo_loss_function=hypo_loss_function,
        hypo_loss_weight=args.hypo_loss_weight,
        hypo_compactness_weight=(
            args.hypo_compactness_weight
        ),
        update_hypo_prototypes=update_hypo_prototypes,
    )
    gap_prior_regularization = outputs["fusion_logits"].sum() * 0.0
    if model.experiment_mode in BILEVEL_ALL_GAP_MODES:
        gap_prior_regularization = (
            model.circular_angle_learner.gap_prior_regularization()
        )
        if args.all_gap_learning_source == "training":
            losses["total_loss"] = (
                losses["total_loss"]
                + args.bilevel_gap_prior_weight
                * gap_prior_regularization
            )
    losses["gap_prior_regularization"] = (
        gap_prior_regularization
    )
    return outputs, losses


def _bilevel_outer_classification_loss(
    outputs,
    batch,
    class_weights,
    args,
):
    fusion_ce = masked_weighted_cross_entropy(
        outputs["fusion_logits"],
        batch["labels"],
        batch["utterance_mask"],
        class_weights,
    )
    confusion_margin = masked_confusion_classification_margin(
        outputs["fusion_cosine_scores"],
        batch["labels"],
        batch["utterance_mask"],
        margin=args.confusion_classification_margin,
    )
    return (
        fusion_ce
        + args.bilevel_outer_confusion_weight * confusion_margin
    )


def bilevel_angle_step(
    model,
    training_batch,
    validation_batch,
    class_weights,
    circular_loss_function,
    args,
    angle_optimizer,
):
    """Approximate a one-step validation hypergradient with DARTS HVP."""
    if model.experiment_mode not in BILEVEL_ANGLE_MODES:
        raise ValueError(
            "bilevel_angle_step requires a bilevel experiment mode"
        )
    if not uses_validation_gap_learning(args):
        raise ValueError(
            "bilevel_angle_step requires validation gap learning"
        )
    learner = model.circular_angle_learner
    if learner is None:
        raise ValueError("bilevel mode requires an angle learner")
    angle_parameters = tuple(learner.parameters())
    if len(angle_parameters) != 1:
        raise ValueError(
            "bilevel mode requires one raw-gap parameter tensor"
        )
    angle_parameter = angle_parameters[0]
    angle_ids = {id(parameter) for parameter in angle_parameters}
    weight_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in angle_ids
    )
    if not weight_parameters:
        raise ValueError("model has no non-angle parameters")

    was_training = model.training
    model.eval()
    gaps_before = torch.rad2deg(
        learner.normalized_gaps()
    ).detach()
    representative_gap_before = float(
        (
            gaps_before[0]
            if model.experiment_mode in BILEVEL_SHARED_GAP_MODES
            else gaps_before.min()
        ).cpu().item()
    )
    gap_prior_regularization = angle_parameter.new_zeros(())
    try:
        validation_outputs = _model_forward(
            model, validation_batch
        )
        outer_loss = _bilevel_outer_classification_loss(
            validation_outputs,
            validation_batch,
            class_weights,
            args,
        )
        validation_vector = torch.autograd.grad(
            outer_loss,
            weight_parameters,
            allow_unused=True,
        )
        vector_norm_squared = outer_loss.new_zeros(())
        for gradient in validation_vector:
            if gradient is not None:
                vector_norm_squared = (
                    vector_norm_squared
                    + gradient.detach().pow(2).sum()
                )
        vector_norm = torch.sqrt(vector_norm_squared)
        if (
            not torch.isfinite(vector_norm)
            or float(vector_norm.detach().cpu().item()) <= 0.0
        ):
            raise RuntimeError(
                "bilevel validation gradient has zero or nonfinite norm"
            )
        epsilon = (
            float(args.bilevel_hvp_radius)
            / float(vector_norm.detach().cpu().item())
        )

        def perturb(scale):
            with torch.no_grad():
                for parameter, gradient in zip(
                    weight_parameters, validation_vector
                ):
                    if gradient is not None:
                        parameter.add_(
                            gradient.detach(),
                            alpha=float(scale),
                        )

        perturbation_state = 0
        perturb(epsilon)
        perturbation_state = 1
        try:
            _, positive_losses = _compute_batch_losses(
                model,
                training_batch,
                class_weights,
                circular_loss_function,
                args,
            )
            positive_gradient = torch.autograd.grad(
                positive_losses["total_loss"],
                angle_parameter,
            )[0]

            perturb(-2.0 * epsilon)
            perturbation_state = -1
            _, negative_losses = _compute_batch_losses(
                model,
                training_batch,
                class_weights,
                circular_loss_function,
                args,
            )
            negative_gradient = torch.autograd.grad(
                negative_losses["total_loss"],
                angle_parameter,
            )[0]
        finally:
            if perturbation_state == 1:
                perturb(-epsilon)
            elif perturbation_state == -1:
                perturb(epsilon)

        hypergradient = (
            -float(args.bilevel_inner_step_size)
            * (positive_gradient - negative_gradient)
            / (2.0 * epsilon)
        )
        if model.experiment_mode in BILEVEL_ALL_GAP_MODES:
            gap_prior_regularization = (
                learner.gap_prior_regularization()
            )
            if args.bilevel_gap_prior_weight > 0.0:
                prior_gradient = torch.autograd.grad(
                    args.bilevel_gap_prior_weight
                    * gap_prior_regularization,
                    angle_parameter,
                )[0]
                hypergradient = hypergradient + prior_gradient
        if not torch.isfinite(hypergradient).all():
            raise RuntimeError("bilevel hypergradient is nonfinite")
        clip = float(args.bilevel_angle_gradient_clip)
        if clip > 0.0:
            hypergradient = hypergradient.clamp(
                min=-clip, max=clip
            )
        angle_optimizer.zero_grad()
        angle_parameter.grad = hypergradient.detach().clone()
        angle_optimizer.step()
    finally:
        model.train(was_training)

    gaps_after = torch.rad2deg(
        learner.normalized_gaps()
    ).detach()
    representative_gap_after = float(
        (
            gaps_after[0]
            if model.experiment_mode in BILEVEL_SHARED_GAP_MODES
            else gaps_after.min()
        ).cpu().item()
    )
    hypergradient_norm = float(
        torch.linalg.vector_norm(hypergradient.detach()).cpu().item()
    )
    gap_change = float(
        torch.linalg.vector_norm(gaps_after - gaps_before)
        .cpu()
        .item()
    )
    return {
        "bilevel_outer_classification_loss": float(
            outer_loss.detach().cpu().item()
        ),
        "bilevel_gap_prior_regularization": float(
            gap_prior_regularization.detach().cpu().item()
        ),
        "bilevel_hypergradient": float(
            (
                hypergradient.detach().reshape(-1)[0]
                if hypergradient.numel() == 1
                else torch.linalg.vector_norm(hypergradient.detach())
            )
            .cpu()
            .item()
        ),
        "bilevel_hypergradient_norm": hypergradient_norm,
        "bilevel_gap_before_degrees": representative_gap_before,
        "bilevel_gap_after_degrees": representative_gap_after,
        "bilevel_gap_change_l2_degrees": gap_change,
        "bilevel_updates": 1.0,
    }


def run_epoch(
    model,
    dataloader,
    device,
    class_weights,
    circular_loss_function,
    args,
    optimizer=None,
    collect_outputs=False,
    bilevel_angle_optimizer=None,
    bilevel_validation_batches=None,
    hypo_loss_function=None,
):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in LOSS_NAMES}
    total_utterances = 0
    all_labels = []
    all_predictions = []
    prediction_rows = []
    text_features = []
    audio_features = []
    visual_features = []
    fusion_features = []
    projected_embeddings = []
    text_projected_embeddings = []
    audio_projected_embeddings = []
    visual_projected_embeddings = []
    bilevel_totals = {
        name: 0.0 for name in BILEVEL_METRIC_NAMES
    }

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if training:
            if bilevel_angle_optimizer is not None:
                if bilevel_validation_batches is None:
                    raise ValueError(
                        "bilevel training requires validation batches"
                    )
                validation_batch = move_batch_to_device(
                    next(bilevel_validation_batches),
                    device,
                )
                bilevel_metrics = bilevel_angle_step(
                    model,
                    batch,
                    validation_batch,
                    class_weights,
                    circular_loss_function,
                    args,
                    bilevel_angle_optimizer,
                )
                for name in BILEVEL_METRIC_NAMES:
                    bilevel_totals[name] += bilevel_metrics[name]
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs, losses = _compute_batch_losses(
                model,
                batch,
                class_weights,
                circular_loss_function,
                args,
                hypo_loss_function=hypo_loss_function,
                update_hypo_prototypes=training,
            )
            if training:
                losses["total_loss"].backward()
                optimizer.step()

        valid = batch["utterance_mask"].reshape(-1) > 0
        valid_count = int(valid.sum().item())
        total_utterances += valid_count
        for name in LOSS_NAMES:
            totals[name] += (
                float(losses[name].detach().cpu().item()) * valid_count
            )

        flat_logits = outputs["fusion_logits"].reshape(
            -1, outputs["fusion_logits"].size(-1)
        )
        valid_probabilities = F.softmax(
            flat_logits[valid], dim=-1
        ).detach()
        valid_predictions = torch.argmax(
            valid_probabilities, dim=-1
        )
        valid_labels = batch["labels"].reshape(-1)[valid]
        all_predictions.append(valid_predictions.cpu().numpy())
        all_labels.append(valid_labels.detach().cpu().numpy())

        if collect_outputs:
            flat_text = outputs["text_representation"].reshape(
                -1, outputs["text_representation"].size(-1)
            )
            flat_audio = outputs["audio_representation"].reshape(
                -1, outputs["audio_representation"].size(-1)
            )
            flat_visual = outputs["visual_representation"].reshape(
                -1, outputs["visual_representation"].size(-1)
            )
            flat_fusion = outputs["fusion_features"].reshape(
                -1, outputs["fusion_features"].size(-1)
            )
            text_features.append(
                flat_text[valid].detach().cpu().numpy()
            )
            audio_features.append(
                flat_audio[valid].detach().cpu().numpy()
            )
            visual_features.append(
                flat_visual[valid].detach().cpu().numpy()
            )
            fusion_features.append(
                flat_fusion[valid].detach().cpu().numpy()
            )
            if outputs["embeddings"] is not None:
                flat_embeddings = outputs["embeddings"].reshape(
                    -1, outputs["embeddings"].size(-1)
                )
                projected_embeddings.append(
                    flat_embeddings[valid].detach().cpu().numpy()
                )
            for output_name, destination in (
                ("text_embeddings", text_projected_embeddings),
                ("audio_embeddings", audio_projected_embeddings),
                ("visual_embeddings", visual_projected_embeddings),
            ):
                modality_embeddings = outputs[output_name]
                if modality_embeddings is not None:
                    flat_modality_embeddings = modality_embeddings.reshape(
                        -1, modality_embeddings.size(-1)
                    )
                    destination.append(
                        flat_modality_embeddings[valid]
                        .detach()
                        .cpu()
                        .numpy()
                    )

            probability_tensor = F.softmax(
                outputs["fusion_logits"], dim=-1
            ).detach().cpu()
            label_tensor = batch["labels"].detach().cpu()
            for dialogue_index, video_id in enumerate(
                batch["video_ids"]
            ):
                length = int(
                    batch["utterance_mask"][dialogue_index]
                    .sum()
                    .item()
                )
                utterance_ids = batch["utterance_ids"][dialogue_index]
                sentences = batch["sentences"][dialogue_index]
                for utterance_index in range(length):
                    probabilities = probability_tensor[
                        dialogue_index, utterance_index
                    ].numpy()
                    true_label = int(
                        label_tensor[dialogue_index, utterance_index]
                    )
                    predicted_label = int(np.argmax(probabilities))
                    row = {
                        "video_id": video_id,
                        "utterance_id": utterance_ids[utterance_index],
                        "utterance_index": utterance_index,
                        "sentence": sentences[utterance_index],
                        "true_label": true_label,
                        "true_emotion": ID2EMOTION[true_label],
                        "predicted_label": predicted_label,
                        "predicted_emotion": ID2EMOTION[predicted_label],
                    }
                    for class_id, emotion in enumerate(EMOTION_NAMES):
                        row["probability_{}".format(emotion)] = float(
                            probabilities[class_id]
                        )
                    prediction_rows.append(row)

    if total_utterances == 0:
        raise RuntimeError("dataloader produced no valid utterances")
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)
    result = {
        name: totals[name] / total_utterances
        for name in LOSS_NAMES
    }
    result.update(classification_metrics(labels, predictions))
    result.update(confusion_pair_metrics(labels, predictions))
    update_count = bilevel_totals["bilevel_updates"]
    if update_count > 0:
        for name in BILEVEL_METRIC_NAMES:
            result[name] = (
                bilevel_totals[name]
                if name == "bilevel_updates"
                else bilevel_totals[name] / update_count
            )
    else:
        for name in BILEVEL_METRIC_NAMES:
            result[name] = None
    result["hypo_initialized_classes"] = (
        None
        if hypo_loss_function is None
        else hypo_loss_function.initialized_classes
    )
    result["hypo_prototype_coverage"] = (
        None
        if hypo_loss_function is None
        else hypo_loss_function.prototype_coverage
    )
    scale = model.effective_cosine_scale
    result["cosine_scale"] = (
        None if scale is None else float(scale.detach().cpu().item())
    )
    for modality, modality_scale in (
        model.effective_unimodal_cosine_scales.items()
    ):
        result["{}_cosine_scale".format(modality)] = (
            None
            if modality_scale is None
            else float(modality_scale.detach().cpu().item())
        )

    if collect_outputs:
        result["labels_array"] = labels
        result["predictions_array"] = predictions
        result["prediction_rows"] = prediction_rows
        result["text_features_array"] = np.concatenate(
            text_features, axis=0
        )
        result["audio_features_array"] = np.concatenate(
            audio_features, axis=0
        )
        result["visual_features_array"] = np.concatenate(
            visual_features, axis=0
        )
        result["fusion_features_array"] = np.concatenate(
            fusion_features, axis=0
        )
        result["embeddings_array"] = (
            np.concatenate(projected_embeddings, axis=0)
            if projected_embeddings
            else None
        )
        result["text_embeddings_array"] = (
            np.concatenate(text_projected_embeddings, axis=0)
            if text_projected_embeddings
            else None
        )
        result["audio_embeddings_array"] = (
            np.concatenate(audio_projected_embeddings, axis=0)
            if audio_projected_embeddings
            else None
        )
        result["visual_embeddings_array"] = (
            np.concatenate(visual_projected_embeddings, axis=0)
            if visual_projected_embeddings
            else None
        )
    return result


def collect_split_with_residual_diagnostics(
    model,
    dataloader,
    device,
    class_weights,
    circular_loss_function,
    args,
    hypo_loss_function=None,
):
    model.enable_spherical_residual_diagnostics(
        enabled=True,
        reset=True,
    )
    try:
        result = run_epoch(
            model,
            dataloader,
            device,
            class_weights,
            circular_loss_function,
            args,
            collect_outputs=True,
            hypo_loss_function=hypo_loss_function,
        )
        diagnostics = model.spherical_residual_diagnostics()
    finally:
        model.enable_spherical_residual_diagnostics(
            enabled=False,
            reset=False,
        )
    return result, diagnostics


def emotion_to_sentiment(emotion_labels):
    emotion_labels = np.asarray(emotion_labels, dtype=np.int64)
    if emotion_labels.size:
        if emotion_labels.min() < 0 or emotion_labels.max() > 5:
            raise ValueError("emotion labels must be in [0, 5]")
    mapping = np.asarray([2, 0, 1, 0, 2, 0], dtype=np.int64)
    return mapping[emotion_labels]


def save_feature_npz(path, result):
    rows = result.get("prediction_rows")
    if rows is None:
        raise ValueError("feature export requires collected prediction rows")
    labels = np.asarray(result["labels_array"], dtype=np.int64)
    predictions = np.asarray(
        result["predictions_array"], dtype=np.int64
    )
    if len(rows) != labels.shape[0]:
        raise ValueError(
            "metadata row count does not match exported utterances"
        )
    export = {
        "labels_emo": labels,
        "preds_emo": predictions,
        "labels_sen": emotion_to_sentiment(labels),
        "preds_sen": emotion_to_sentiment(predictions),
        "dialogue_ids": np.asarray(
            [row["video_id"] for row in rows]
        ),
        "utterance_indices": np.asarray(
            [row["utterance_index"] for row in rows],
            dtype=np.int64,
        ),
        "utterance_ids": np.asarray(
            [row["utterance_id"] for row in rows]
        ),
        "sentences": np.asarray([row["sentence"] for row in rows]),
        "feature_l": np.asarray(
            result["text_features_array"], dtype=np.float32
        ),
        "feature_v": np.asarray(
            result["visual_features_array"], dtype=np.float32
        ),
        "feature_a": np.asarray(
            result["audio_features_array"], dtype=np.float32
        ),
        "feature_fusion": np.asarray(
            result["fusion_features_array"], dtype=np.float32
        ),
    }
    if result["embeddings_array"] is not None:
        export["feature_embedding"] = np.asarray(
            result["embeddings_array"], dtype=np.float32
        )
    for export_name, result_name in (
        ("feature_l_embedding", "text_embeddings_array"),
        ("feature_a_embedding", "audio_embeddings_array"),
        ("feature_v_embedding", "visual_embeddings_array"),
    ):
        values = result.get(result_name)
        if values is not None:
            export[export_name] = np.asarray(values, dtype=np.float32)
    expected_rows = labels.shape[0]
    for name, values in export.items():
        if values.shape[0] != expected_rows:
            raise ValueError(
                "{} has {} rows; expected {}".format(
                    name, values.shape[0], expected_rows
                )
            )
    np.savez_compressed(path, **export)
    return path


def save_empty_feature_npz(path, model):
    """Write a schema-compatible empty validation export."""
    empty_int = np.empty((0,), dtype=np.int64)
    empty_text = np.empty((0,), dtype=str)
    export = {
        "labels_emo": empty_int,
        "preds_emo": empty_int.copy(),
        "labels_sen": empty_int.copy(),
        "preds_sen": empty_int.copy(),
        "dialogue_ids": empty_text,
        "utterance_indices": empty_int.copy(),
        "utterance_ids": empty_text.copy(),
        "sentences": empty_text.copy(),
        "feature_l": np.empty(
            (0, model.hidden_dim), dtype=np.float32
        ),
        "feature_v": np.empty(
            (0, model.hidden_dim), dtype=np.float32
        ),
        "feature_a": np.empty(
            (0, model.hidden_dim), dtype=np.float32
        ),
        "feature_fusion": np.empty(
            (0, model.hidden_dim), dtype=np.float32
        ),
    }
    if model.fusion_projector is not None:
        export["feature_embedding"] = np.empty(
            (0, model.embedding_dim), dtype=np.float32
        )
    if model.text_projector is not None:
        for name in (
            "feature_l_embedding",
            "feature_a_embedding",
            "feature_v_embedding",
        ):
            export[name] = np.empty(
                (0, model.embedding_dim), dtype=np.float32
            )
    np.savez_compressed(path, **export)
    return path


def public_metrics(result):
    keys = list(LOSS_NAMES) + [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "macro_recall",
        *CONFUSION_PAIR_METRIC_NAMES,
        *BILEVEL_METRIC_NAMES,
        "cosine_scale",
        "text_cosine_scale",
        "audio_cosine_scale",
        "visual_cosine_scale",
        "hypo_initialized_classes",
        "hypo_prototype_coverage",
    ]
    return {key: result.get(key) for key in keys}


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=True)


def save_hypo_artifacts(
    run_dir,
    hypo_loss_function,
    args,
    selected_epoch,
):
    if hypo_loss_function is None:
        return None
    missing = hypo_loss_function.missing_class_ids()
    if missing:
        raise RuntimeError(
            "cannot export incomplete HYPO prototypes; missing "
            "class IDs: {}".format(missing)
        )
    prototypes = F.normalize(
        hypo_loss_function.prototypes.detach(),
        p=2,
        dim=-1,
        eps=1e-8,
    ).cpu().numpy()
    initialized = (
        hypo_loss_function.initialized.detach().cpu().numpy()
    )
    update_counts = (
        hypo_loss_function.update_counts.detach().cpu().numpy()
    )
    similarities = np.clip(
        np.matmul(prototypes, prototypes.T), -1.0, 1.0
    )
    angular_distances = np.degrees(np.arccos(similarities))
    np.savez_compressed(
        os.path.join(run_dir, "hypo_prototypes.npz"),
        prototypes=prototypes,
        initialized=initialized,
        update_counts=update_counts,
    )
    payload = {
        "selected_epoch": int(selected_epoch),
        "num_classes": int(hypo_loss_function.num_classes),
        "embedding_dim": int(hypo_loss_function.embedding_dim),
        "prototype_norms": np.linalg.norm(
            prototypes, axis=1
        ).tolist(),
        "prototype_cosine_similarity": similarities.tolist(),
        "prototype_angular_distance_degrees": (
            angular_distances.tolist()
        ),
        "initialized": initialized.tolist(),
        "initialized_classes": (
            hypo_loss_function.initialized_classes
        ),
        "prototype_coverage": (
            hypo_loss_function.prototype_coverage
        ),
        "update_counts": update_counts.tolist(),
        "hypo_loss_weight": float(args.hypo_loss_weight),
        "hypo_compactness_weight": float(
            args.hypo_compactness_weight
        ),
        "hypo_temperature": float(args.hypo_temperature),
        "hypo_prototype_momentum": float(
            args.hypo_prototype_momentum
        ),
    }
    write_json(
        os.path.join(run_dir, "hypo_geometry.json"),
        payload,
    )
    return payload


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_epoch_metrics(path, rows):
    flattened = []
    for row in rows:
        item = {"epoch": row["epoch"], "seconds": row["seconds"]}
        for split in ("training", "validation", "testing"):
            if split not in row:
                continue
            for key, value in row[split].items():
                item["{}_{}".format(split, key)] = value
        flattened.append(item)
    write_rows(path, flattened)


def is_better_validation(
    candidate_f1,
    candidate_ce,
    best_f1,
    best_ce,
    tolerance=1e-12,
):
    if candidate_f1 > best_f1 + tolerance:
        return True
    if abs(candidate_f1 - best_f1) <= tolerance:
        return candidate_ce < best_ce - tolerance
    return False


def is_better_selection(
    candidate_f1,
    candidate_ce,
    best_f1,
    best_ce,
    selection_protocol,
    tolerance=1e-12,
):
    if selection_protocol == "test":
        return (
            round(candidate_f1, 2)
            > round(best_f1, 2) + tolerance
        )
    return is_better_validation(
        candidate_f1,
        candidate_ce,
        best_f1,
        best_ce,
        tolerance=tolerance,
    )


def experiment_directory_name(
    mode,
    circular_weight,
    circular_geometry="equal",
    angle_weight=0.0,
    selection_protocol="validation",
    sdt_residual_update="standard",
    spherical_attention_alpha_init=0.1,
    spherical_mlp_alpha_init=0.1,
    minimum_confusion_gap_degrees=75.0,
    confusion_gap_weight=0.0,
    confused_cse_pair_weight=1.0,
    confusion_classification_margin=0.1,
    confusion_classification_weight=0.0,
    bilevel_gap_minimum_degrees=70.0,
    bilevel_gap_maximum_degrees=110.0,
    bilevel_gap_initial_degrees=90.0,
    bilevel_angle_learning_rate=0.001,
    bilevel_outer_confusion_weight=0.1,
    bilevel_all_gaps_initialization="equal",
    bilevel_minimum_class_gap_degrees=20.0,
    bilevel_gap_prior_weight=0.01,
    all_gap_learning_source="validation",
    hypo_loss_weight=0.1,
    hypo_compactness_weight=2.0,
    hypo_temperature=0.1,
    hypo_prototype_momentum=0.95,
):
    if mode in HYPO_MODES:
        condition = "{}_lambda_{}_w{}_tau{}_pm{}".format(
            mode,
            format(float(hypo_loss_weight), "g"),
            format(float(hypo_compactness_weight), "g"),
            format(float(hypo_temperature), "g"),
            format(float(hypo_prototype_momentum), "g"),
        )
    elif mode in BILEVEL_ALL_GAP_MODES:
        condition = (
            "{}_lambda_{}_init_{}_mingap_{}_prior_{}_pair_{}_"
            "clsmargin_{}_clsweight_{}"
        ).format(
            mode,
            format(float(circular_weight), "g"),
            bilevel_all_gaps_initialization,
            format(
                float(bilevel_minimum_class_gap_degrees), "g"
            ),
            format(float(bilevel_gap_prior_weight), "g"),
            format(float(confused_cse_pair_weight), "g"),
            format(float(confusion_classification_margin), "g"),
            format(float(confusion_classification_weight), "g"),
        )
        if all_gap_learning_source == "validation":
            condition += "_anglelr_{}_outerconf_{}".format(
                format(float(bilevel_angle_learning_rate), "g"),
                format(float(bilevel_outer_confusion_weight), "g"),
            )
        else:
            condition += "_source_{}".format(
                all_gap_learning_source
            )
    elif mode in BILEVEL_SHARED_GAP_MODES:
        condition = (
            "{}_lambda_{}_range_{}-{}_init_{}_pair_{}_"
            "clsmargin_{}_clsweight_{}_anglelr_{}_outerconf_{}"
        ).format(
            mode,
            format(float(circular_weight), "g"),
            format(float(bilevel_gap_minimum_degrees), "g"),
            format(float(bilevel_gap_maximum_degrees), "g"),
            format(float(bilevel_gap_initial_degrees), "g"),
            format(float(confused_cse_pair_weight), "g"),
            format(float(confusion_classification_margin), "g"),
            format(float(confusion_classification_weight), "g"),
            format(float(bilevel_angle_learning_rate), "g"),
            format(float(bilevel_outer_confusion_weight), "g"),
        )
    elif mode in CONFUSION_MARGIN_MODES:
        condition = (
            "{}_{}_lambda_{}_mingap_{}_pair_{}_"
            "clsmargin_{}_clsweight_{}"
        ).format(
            mode,
            circular_geometry,
            format(float(circular_weight), "g"),
            format(float(minimum_confusion_gap_degrees), "g"),
            format(float(confused_cse_pair_weight), "g"),
            format(float(confusion_classification_margin), "g"),
            format(float(confusion_classification_weight), "g"),
        )
    elif mode in CONFUSION_GAP_MODES:
        condition = (
            "{}_{}_lambda_{}_angle_{}_mingap_{}_gap_{}"
        ).format(
            mode,
            circular_geometry,
            format(float(circular_weight), "g"),
            format(float(angle_weight), "g"),
            format(float(minimum_confusion_gap_degrees), "g"),
            format(float(confusion_gap_weight), "g"),
        )
    elif mode in LEARNABLE_ANGLE_MODES:
        condition = "{}_{}_lambda_{}_angle_{}".format(
            mode,
            circular_geometry,
            format(float(circular_weight), "g"),
            format(float(angle_weight), "g"),
        )
    elif mode in CIRCULAR_CSE_MODES:
        geometry_suffix = (
            "" if circular_geometry == "equal"
            else "_{}".format(circular_geometry)
        )
        condition = "{}{}_lambda_{}".format(
            mode,
            geometry_suffix,
            format(float(circular_weight), "g"),
        )
    else:
        condition = mode
    if sdt_residual_update == "spherical":
        condition += "_spherical_residual_a{}_m{}".format(
            format(float(spherical_attention_alpha_init), "g"),
            format(float(spherical_mlp_alpha_init), "g"),
        )
    if selection_protocol == "test":
        condition += "_test_selected"
    return condition


def current_angle_state(model, fixed_class_angles):
    learnable_state = model.current_circular_angle_state()
    if learnable_state is None:
        return {
            "learnable": False,
            "prior_angles": fixed_class_angles,
            "angles": fixed_class_angles,
            "gaps": None,
            "offsets": torch.zeros_like(fixed_class_angles),
            "regularization": fixed_class_angles.sum() * 0.0,
            "raw_gaps": None,
            "circle_order": torch.tensor(
                CIRCLE_ORDER,
                device=fixed_class_angles.device,
                dtype=torch.long,
            ),
        }
    learner = model.circular_angle_learner
    return {
        "learnable": True,
        "prior_angles": learner.prior_angles,
        "angles": learnable_state["angles"],
        "gaps": learnable_state["gaps"],
        "offsets": learnable_state["offsets"],
        "regularization": learnable_state["regularization"],
        "raw_gaps": learner.raw_gaps,
        "circle_order": learner.circle_order,
    }


def angle_state_payload(model, fixed_class_angles):
    state = current_angle_state(model, fixed_class_angles)

    def values(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().tolist()

    def degrees(tensor):
        if tensor is None:
            return None
        return (
            tensor.detach().cpu() * (180.0 / np.pi)
        ).tolist()

    angles = state["angles"]
    return {
        "learnable": state["learnable"],
        "circle_order": values(state["circle_order"]),
        "prior_angles_radians": values(state["prior_angles"]),
        "prior_angles_degrees": degrees(state["prior_angles"]),
        "class_angles_radians": values(angles),
        "class_angles_degrees": degrees(angles),
        "angle_offsets_radians": values(state["offsets"]),
        "angle_offsets_degrees": degrees(state["offsets"]),
        "normalized_gaps_radians": values(state["gaps"]),
        "normalized_gaps_degrees": degrees(state["gaps"]),
        "raw_gaps": values(state["raw_gaps"]),
        "angle_regularization": float(
            state["regularization"].detach().cpu().item()
        ),
        "target_similarity": values(
            build_target_similarity(angles)
        ),
    }


def angle_history_row(
    epoch,
    model,
    fixed_class_angles,
    minimum_confusion_gap_degrees=None,
):
    state = current_angle_state(model, fixed_class_angles)
    if not state["learnable"]:
        return None
    angles = (
        state["angles"].detach().cpu().numpy()
        * (180.0 / np.pi)
    )
    gaps = (
        state["gaps"].detach().cpu().numpy()
        * (180.0 / np.pi)
    )
    row = {
        "epoch": int(epoch),
        "angle_regularization": float(
            state["regularization"].detach().cpu().item()
        ),
    }
    for class_id, emotion in ID2EMOTION.items():
        row["angle_{}_degrees".format(emotion)] = float(
            angles[class_id]
        )
    order = state["circle_order"].detach().cpu().tolist()
    for position, class_id in enumerate(order):
        source = ID2EMOTION[int(class_id)]
        target = ID2EMOTION[int(order[(position + 1) % len(order)])]
        row["gap_{}_to_{}_degrees".format(source, target)] = float(
            gaps[position]
        )
    if minimum_confusion_gap_degrees is not None:
        gap_payload = confusion_gap_payload(
            model,
            fixed_class_angles,
            minimum_confusion_gap_degrees,
        )
        row["minimum_confusion_gap_degrees"] = (
            gap_payload["minimum_confusion_gap_degrees"]
        )
        row["confusion_gap_regularization"] = gap_payload[
            "confusion_gap_regularization"
        ]
        for name, gap in gap_payload[
            "confusion_pair_gaps_degrees"
        ].items():
            row["confusion_gap_{}_degrees".format(name)] = gap
    return row


def confusion_gap_payload(
    model,
    fixed_class_angles,
    minimum_gap_degrees,
):
    state = current_angle_state(model, fixed_class_angles)
    angles = state["angles"]
    distances = circular_pair_distances(
        angles,
        IEMOCAP_CONFUSION_PAIRS,
    )
    penalty = minimum_confusion_gap_regularization(
        angles,
        minimum_gap_degrees=minimum_gap_degrees,
        pairs=IEMOCAP_CONFUSION_PAIRS,
    )
    pair_gaps = {}
    for pair, distance in zip(
        IEMOCAP_CONFUSION_PAIRS,
        distances.detach().cpu().tolist(),
    ):
        first_id, second_id = pair
        name = "{}_{}".format(
            ID2EMOTION[first_id],
            ID2EMOTION[second_id],
        )
        pair_gaps[name] = float(math.degrees(distance))
    return {
        "minimum_confusion_gap_degrees": float(
            minimum_gap_degrees
        ),
        "confusion_pairs": [
            {
                "first_id": int(first_id),
                "first_emotion": ID2EMOTION[first_id],
                "second_id": int(second_id),
                "second_emotion": ID2EMOTION[second_id],
            }
            for first_id, second_id in IEMOCAP_CONFUSION_PAIRS
        ],
        "confusion_pair_gaps_degrees": pair_gaps,
        "confusion_gap_regularization": float(
            penalty.detach().cpu().item()
        ),
    }


def bilevel_gap_payload(model, args):
    if model.experiment_mode not in BILEVEL_ANGLE_MODES:
        return None
    learner = model.circular_angle_learner
    gaps = torch.rad2deg(
        learner.normalized_gaps().detach().cpu()
    ).tolist()
    order = learner.circle_order.detach().cpu().tolist()
    named_gaps = {}
    for position, class_id in enumerate(order):
        next_class_id = order[(position + 1) % len(order)]
        named_gaps[
            "{}_to_{}".format(
                ID2EMOTION[int(class_id)],
                ID2EMOTION[int(next_class_id)],
            )
        ] = float(gaps[position])
    common = {
        "normalized_gaps_degrees": [
            float(value) for value in gaps
        ],
        "named_gaps_degrees": named_gaps,
    }
    if model.experiment_mode in BILEVEL_ALL_GAP_MODES:
        prior_penalty = float(
            learner.gap_prior_regularization()
            .detach()
            .cpu()
            .item()
        )
        if args.all_gap_learning_source == "training":
            optimization = {
                "optimizer": "main_adam_zero_weight_decay",
                "angle_learning_rate": float(args.lr),
                "inner_step_size": None,
                "hvp_radius": None,
                "outer_confusion_weight": None,
                "angle_gradient_clip": None,
            }
        else:
            optimization = {
                "optimizer": "separate_outer_adam_zero_weight_decay",
                "angle_learning_rate": float(
                    args.bilevel_angle_learning_rate
                ),
                "inner_step_size": float(
                    args.bilevel_inner_step_size
                ),
                "hvp_radius": float(args.bilevel_hvp_radius),
                "outer_confusion_weight": float(
                    args.bilevel_outer_confusion_weight
                ),
                "angle_gradient_clip": float(
                    args.bilevel_angle_gradient_clip
                ),
            }
        return {
            "parameterization": (
                "minimum_floor_plus_softmax_all_gap_allocation"
            ),
            "learning_source": args.all_gap_learning_source,
            "initialization": (
                args.bilevel_all_gaps_initialization
            ),
            "minimum_class_gap_degrees": float(
                args.bilevel_minimum_class_gap_degrees
            ),
            "gap_prior_weight": float(
                args.bilevel_gap_prior_weight
            ),
            "gap_prior_regularization": prior_penalty,
            "selected_minimum_gap_degrees": float(min(gaps)),
            "selected_maximum_gap_degrees": float(max(gaps)),
            "raw_parameters": [
                float(value)
                for value in learner.raw_gaps.detach().cpu().tolist()
            ],
            **common,
            **optimization,
        }
    optimization = {
        "optimizer": "separate_outer_adam_zero_weight_decay",
        "angle_learning_rate": float(
            args.bilevel_angle_learning_rate
        ),
        "inner_step_size": float(args.bilevel_inner_step_size),
        "hvp_radius": float(args.bilevel_hvp_radius),
        "outer_confusion_weight": float(
            args.bilevel_outer_confusion_weight
        ),
        "angle_gradient_clip": float(
            args.bilevel_angle_gradient_clip
        ),
    }
    return {
        "parameterization": (
            "shared_confusion_gap_with_balanced_remaining_gaps"
        ),
        "minimum_degrees": float(
            args.bilevel_gap_minimum_degrees
        ),
        "maximum_degrees": float(
            args.bilevel_gap_maximum_degrees
        ),
        "initial_degrees": float(
            args.bilevel_gap_initial_degrees
        ),
        "selected_confusion_gap_degrees": float(gaps[0]),
        "selected_remaining_gap_degrees": float(gaps[1]),
        "normalized_gaps_degrees": [float(value) for value in gaps],
        "raw_parameter": float(
            learner.raw_gaps.detach().cpu().item()
        ),
        **common,
        **optimization,
    }


def spherical_residual_gate_payload(model):
    return {
        name: float(alpha.detach().cpu().item())
        for name, alpha in model.spherical_residual_gate_state().items()
    }


def spherical_residual_history_row(epoch, model):
    gates = spherical_residual_gate_payload(model)
    if not gates:
        return None
    row = {"epoch": int(epoch)}
    for name, alpha in gates.items():
        column = "{}_alpha".format(
            name.replace(".", "_")
        )
        row[column] = alpha
    return row


def build_optimizer(
    model,
    learning_rate,
    weight_decay,
    exclude_angle_parameters=False,
):
    no_decay_parameters = []
    excluded_parameter_ids = set()
    if model.circular_angle_learner is not None:
        angle_parameters = tuple(
            model.circular_angle_learner.parameters()
        )
        if exclude_angle_parameters:
            excluded_parameter_ids.update(
                id(parameter) for parameter in angle_parameters
            )
        else:
            no_decay_parameters.extend(angle_parameters)
    no_decay_parameters.extend(
        model.spherical_residual_parameters()
    )
    if not no_decay_parameters and not excluded_parameter_ids:
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    no_decay_parameter_ids = {
        id(parameter) for parameter in no_decay_parameters
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            id(parameter) not in no_decay_parameter_ids
            and id(parameter) not in excluded_parameter_ids
        )
    ]
    return optim.Adam(
        [
            {
                "params": base_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        weight_decay=0.0,
    )


def build_bilevel_angle_optimizer(model, learning_rate):
    if model.experiment_mode not in BILEVEL_ANGLE_MODES:
        return None
    if learning_rate <= 0.0:
        raise ValueError("bilevel angle learning rate must be positive")
    parameters = tuple(model.circular_angle_learner.parameters())
    if len(parameters) != 1:
        raise ValueError(
            "bilevel mode requires one raw-gap parameter tensor"
        )
    return optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    args,
    split_ids,
    epoch,
    validation_metrics=None,
    selection_metrics=None,
    angle_optimizer=None,
    bilevel_metrics=None,
    hypo_loss_function=None,
):
    if args.experiment_mode in HYPO_MODES:
        angle_payload = {
            "circle_order": None,
            "prior_angles_radians": None,
            "class_angles_radians": None,
            "normalized_gaps_radians": None,
            "angle_offsets_radians": None,
            "raw_gaps": None,
            "angle_regularization": None,
        }
        gap_payload = {
            "minimum_confusion_gap_degrees": None,
            "confusion_pairs": None,
            "confusion_pair_gaps_degrees": None,
            "confusion_gap_regularization": None,
        }
        target_similarity = None
    else:
        angles = build_iemocap_angles(
            geometry=args.circular_geometry,
            vad_center=(
                args.vad_center_valence,
                args.vad_center_arousal,
            ),
            minimum_confusion_gap_degrees=(
                args.minimum_confusion_gap_degrees
            ),
        )
        angle_payload = angle_state_payload(model, angles)
        gap_payload = confusion_gap_payload(
            model,
            angles,
            args.minimum_confusion_gap_degrees,
        )
        selected_angles = torch.tensor(
            angle_payload["class_angles_radians"],
            dtype=angles.dtype,
        )
        target_similarity = build_target_similarity(
            selected_angles
        ).tolist()
    payload = {
        "model_state_dict": model.state_dict(),
        "hypo_state_dict": (
            None
            if hypo_loss_function is None
            else hypo_loss_function.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "angle_optimizer_state_dict": (
            None
            if angle_optimizer is None
            else angle_optimizer.state_dict()
        ),
        "config": vars(args).copy(),
        "emotion_mapping": ID2EMOTION,
        "circular_geometry": args.circular_geometry,
        "vad_center": (
            None
            if args.experiment_mode in HYPO_MODES
            else [
                args.vad_center_valence,
                args.vad_center_arousal,
            ]
        ),
        "nrc_vad_anchors": (
            None
            if args.experiment_mode in HYPO_MODES
            else build_iemocap_vad_anchors().tolist()
        ),
        "angle_weight": args.angle_weight,
        "confusion_gap_weight": args.confusion_gap_weight,
        "confused_cse_pair_weight": args.confused_cse_pair_weight,
        "confusion_classification_margin": (
            args.confusion_classification_margin
        ),
        "confusion_classification_weight": (
            args.confusion_classification_weight
        ),
        "bilevel_gap_minimum_degrees": (
            args.bilevel_gap_minimum_degrees
        ),
        "bilevel_gap_maximum_degrees": (
            args.bilevel_gap_maximum_degrees
        ),
        "bilevel_gap_initial_degrees": (
            args.bilevel_gap_initial_degrees
        ),
        "bilevel_all_gaps_initialization": (
            args.bilevel_all_gaps_initialization
        ),
        "bilevel_minimum_class_gap_degrees": (
            args.bilevel_minimum_class_gap_degrees
        ),
        "bilevel_gap_prior_weight": (
            args.bilevel_gap_prior_weight
        ),
        "all_gap_learning_source": (
            args.all_gap_learning_source
        ),
        "bilevel_angle_learning_rate": (
            args.bilevel_angle_learning_rate
        ),
        "bilevel_inner_step_size": args.bilevel_inner_step_size,
        "bilevel_hvp_radius": args.bilevel_hvp_radius,
        "bilevel_outer_confusion_weight": (
            args.bilevel_outer_confusion_weight
        ),
        **gap_payload,
        "circle_order": angle_payload["circle_order"],
        "prior_class_angles": angle_payload[
            "prior_angles_radians"
        ],
        "class_angles": angle_payload["class_angles_radians"],
        "angle_gaps": angle_payload["normalized_gaps_radians"],
        "angle_offsets": angle_payload["angle_offsets_radians"],
        "raw_gaps": angle_payload["raw_gaps"],
        "angle_regularization": angle_payload[
            "angle_regularization"
        ],
        "target_similarity": target_similarity,
        "class_weights": iemocap_class_weights().tolist(),
        "split_ids": split_ids,
        "selected_epoch": epoch,
        "validation_metrics": validation_metrics,
        "selection_protocol": args.selection_protocol,
        "selection_split": (
            "testing"
            if args.selection_protocol == "test"
            else "validation"
        ),
        "selection_metrics": (
            selection_metrics
            if selection_metrics is not None
            else validation_metrics
        ),
        "sdt_residual_update": args.sdt_residual_update,
        "spherical_attention_alpha_init": (
            args.spherical_attention_alpha_init
        ),
        "spherical_mlp_alpha_init": args.spherical_mlp_alpha_init,
        "spherical_residual_gates": (
            spherical_residual_gate_payload(model)
        ),
        "bilevel_geometry": bilevel_gap_payload(model, args),
        "bilevel_metrics": bilevel_metrics,
        "hypo_loss_weight": args.hypo_loss_weight,
        "hypo_compactness_weight": (
            args.hypo_compactness_weight
        ),
        "hypo_temperature": args.hypo_temperature,
        "hypo_prototype_momentum": (
            args.hypo_prototype_momentum
        ),
    }
    if args.experiment_mode in HYPO_MODES:
        payload.update(
            {
                "circular_geometry": None,
                "circle_order": None,
                "prior_class_angles": None,
                "class_angles": None,
                "angle_gaps": None,
                "angle_offsets": None,
                "raw_gaps": None,
                "angle_regularization": None,
                "target_similarity": None,
                "bilevel_geometry": None,
                "vad_center": None,
                "nrc_vad_anchors": None,
                "minimum_confusion_gap_degrees": None,
                "confusion_pairs": None,
                "confusion_pair_gaps_degrees": None,
                "confusion_gap_regularization": None,
            }
        )
    torch.save(payload, path)


def final_classification_details(labels, predictions):
    return {
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(6)),
            target_names=EMOTION_NAMES,
            digits=6,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(6))
        ).tolist(),
    }


def validate_arguments(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.embedding_dim < 2:
        raise ValueError("--embedding-dim must be at least 2")
    if (
        args.experiment_mode not in BILEVEL_ALL_GAP_MODES
        and args.all_gap_learning_source != "validation"
    ):
        raise ValueError(
            "--all-gap-learning-source applies only to "
            "sdt_cse_bilevel_all_gaps"
        )
    if args.selection_protocol not in SELECTION_PROTOCOLS:
        raise ValueError(
            "--selection-protocol must be one of {}".format(
                SELECTION_PROTOCOLS
            )
        )
    if args.sdt_residual_update not in SDT_RESIDUAL_UPDATES:
        raise ValueError(
            "--sdt-residual-update must be one of {}".format(
                SDT_RESIDUAL_UPDATES
            )
        )
    for name in (
        "spherical_attention_alpha_init",
        "spherical_mlp_alpha_init",
    ):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise ValueError(
                "--{} must be strictly between 0 and 1".format(
                    name.replace("_", "-")
                )
            )
    for name in (
        "fusion_ce_weight",
        "unimodal_ce_weight",
        "distillation_weight",
        "circular_weight",
        "angle_weight",
        "confusion_gap_weight",
        "confusion_classification_weight",
        "same_class_margin",
        "bilevel_outer_confusion_weight",
        "bilevel_angle_gradient_clip",
        "bilevel_gap_prior_weight",
        "hypo_loss_weight",
        "hypo_compactness_weight",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "--{} must be finite and nonnegative".format(
                    name.replace("_", "-")
                )
            )
    if (
        not np.isfinite(args.hypo_temperature)
        or args.hypo_temperature <= 0.0
    ):
        raise ValueError(
            "--hypo-temperature must be finite and positive"
        )
    if (
        not np.isfinite(args.hypo_prototype_momentum)
        or args.hypo_prototype_momentum < 0.0
        or args.hypo_prototype_momentum >= 1.0
    ):
        raise ValueError(
            "--hypo-prototype-momentum must be finite and in [0, 1)"
        )
    if args.bilevel_inner_step_size is None:
        args.bilevel_inner_step_size = args.lr
    for name in (
        "bilevel_angle_learning_rate",
        "bilevel_inner_step_size",
        "bilevel_hvp_radius",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "--{} must be finite and positive".format(
                    name.replace("_", "-")
                )
            )
    for name in (
        "bilevel_outer_confusion_weight",
        "bilevel_angle_gradient_clip",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "--{} must be finite and nonnegative".format(
                    name.replace("_", "-")
                )
            )
    if not (
        0.0
        < args.bilevel_gap_minimum_degrees
        < args.bilevel_gap_initial_degrees
        < args.bilevel_gap_maximum_degrees
        < 180.0
    ):
        raise ValueError(
            "bilevel gaps must satisfy 0 < minimum < initial "
            "< maximum < 180"
        )
    if (
        not np.isfinite(args.bilevel_minimum_class_gap_degrees)
        or args.bilevel_minimum_class_gap_degrees < 0.0
        or args.bilevel_minimum_class_gap_degrees >= 60.0
    ):
        raise ValueError(
            "--bilevel-minimum-class-gap-degrees must be finite "
            "and in [0, 60)"
        )
    if (
        not np.isfinite(args.minimum_confusion_gap_degrees)
        or args.minimum_confusion_gap_degrees <= 0.0
        or args.minimum_confusion_gap_degrees > 180.0
    ):
        raise ValueError(
            "--minimum-confusion-gap-degrees must be finite and "
            "in (0, 180]"
        )
    if (
        not np.isfinite(args.confused_cse_pair_weight)
        or args.confused_cse_pair_weight < 1.0
    ):
        raise ValueError(
            "--confused-cse-pair-weight must be finite and at least 1"
        )
    if (
        not np.isfinite(args.confusion_classification_margin)
        or args.confusion_classification_margin < 0.0
        or args.confusion_classification_margin > 2.0
    ):
        raise ValueError(
            "--confusion-classification-margin must be finite and "
            "in [0, 2]"
        )
    if not np.isfinite(args.vad_center_valence):
        raise ValueError("--vad-center-valence must be finite")
    if not np.isfinite(args.vad_center_arousal):
        raise ValueError("--vad-center-arousal must be finite")
    if args.circular_geometry is None:
        if args.experiment_mode in BILEVEL_ALL_GAP_MODES:
            args.circular_geometry = (
                args.bilevel_all_gaps_initialization
            )
        elif args.experiment_mode in CONFUSION_MARGIN_MODES:
            args.circular_geometry = "confusion_separated"
        elif args.experiment_mode in CONFUSION_GAP_MODES:
            args.circular_geometry = "equal"
        elif args.experiment_mode in LEARNABLE_ANGLE_MODES:
            args.circular_geometry = "nrc_vad"
        else:
            args.circular_geometry = "equal"
    if (
        args.experiment_mode in CONFUSION_GAP_MODES
        and args.circular_geometry != "equal"
    ):
        raise ValueError(
            "confusion-gap mode requires --circular-geometry equal"
        )
    if (
        args.experiment_mode in CONFUSION_MARGIN_MODES
        and args.experiment_mode not in BILEVEL_ALL_GAP_MODES
        and args.circular_geometry != "confusion_separated"
    ):
        raise ValueError(
            "confusion-margin mode requires "
            "--circular-geometry confusion_separated"
        )
    if (
        args.experiment_mode in CONFUSION_MARGIN_MODES
        and args.experiment_mode not in BILEVEL_ALL_GAP_MODES
        and args.minimum_confusion_gap_degrees >= 180.0
    ):
        raise ValueError(
            "confusion-margin mode requires "
            "--minimum-confusion-gap-degrees below 180"
        )
    if args.experiment_mode in BILEVEL_ANGLE_MODES:
        if (
            uses_validation_gap_learning(args)
            and args.selection_protocol != "validation"
        ):
            raise ValueError(
                "validation-learned gap mode requires "
                "--selection-protocol validation; use "
                "--all-gap-learning-source training for training-only "
                "gap updates"
            )
        if args.circular_weight <= 0.0:
            raise ValueError(
                "bilevel confusion-gap mode requires positive "
                "--circular-weight"
            )
        if args.experiment_mode in BILEVEL_SHARED_GAP_MODES:
            args.minimum_confusion_gap_degrees = (
                args.bilevel_gap_minimum_degrees
            )
    if (
        args.experiment_mode in BILEVEL_ALL_GAP_MODES
        and args.circular_geometry
        != args.bilevel_all_gaps_initialization
    ):
        raise ValueError(
            "all-gap initialization must match --circular-geometry "
            "when both are specified"
        )
    if args.experiment_mode not in CIRCULAR_CSE_MODES:
        args.circular_weight = 0.0
        args.circular_geometry = (
            None
            if args.experiment_mode in HYPO_MODES
            else "equal"
        )
    if args.experiment_mode not in HYPO_MODES:
        args.hypo_loss_weight = 0.0
    validated_initial_angles = (
        None
        if args.experiment_mode in HYPO_MODES
        else build_iemocap_angles(
            geometry=args.circular_geometry,
            vad_center=(
                args.vad_center_valence,
                args.vad_center_arousal,
            ),
            minimum_confusion_gap_degrees=(
                args.minimum_confusion_gap_degrees
            ),
        )
    )
    if args.experiment_mode in BILEVEL_ALL_GAP_MODES:
        BilevelAllGapsAngles(
            num_classes=6,
            minimum_gap_degrees=(
                args.bilevel_minimum_class_gap_degrees
            ),
            prior_angles=validated_initial_angles,
            circle_order=CIRCLE_ORDER,
        )
    if args.experiment_mode in FUSION_ONLY_MODES:
        args.unimodal_ce_weight = 0.0
        args.distillation_weight = 0.0
    if args.experiment_mode not in LEARNABLE_ANGLE_MODES:
        args.angle_weight = 0.0
    if args.experiment_mode not in CONFUSION_GAP_MODES:
        args.confusion_gap_weight = 0.0
    if args.experiment_mode not in CONFUSION_MARGIN_MODES:
        args.confused_cse_pair_weight = 1.0
        args.confusion_classification_weight = 0.0
    if args.experiment_mode in HYPO_MODES:
        args.circular_weight = 0.0
        args.angle_weight = 0.0
        args.confusion_gap_weight = 0.0
        args.confused_cse_pair_weight = 0.0
        args.confusion_classification_weight = 0.0


def train_and_test(args):
    validate_arguments(args)
    device = resolve_device(args.device, args.gpu_id)
    set_random_seed(args.seed, device.type == "cuda")
    loaders = create_iemocap_loaders(
        feature_path=args.feature_path,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        selection_protocol=args.selection_protocol,
    )

    run_dir = os.path.join(
        os.path.abspath(args.output_dir),
        experiment_directory_name(
            args.experiment_mode,
            args.circular_weight,
            args.circular_geometry,
            args.angle_weight,
            args.selection_protocol,
            args.sdt_residual_update,
            args.spherical_attention_alpha_init,
            args.spherical_mlp_alpha_init,
            args.minimum_confusion_gap_degrees,
            args.confusion_gap_weight,
            args.confused_cse_pair_weight,
            args.confusion_classification_margin,
            args.confusion_classification_weight,
            args.bilevel_gap_minimum_degrees,
            args.bilevel_gap_maximum_degrees,
            args.bilevel_gap_initial_degrees,
            args.bilevel_angle_learning_rate,
            args.bilevel_outer_confusion_weight,
            args.bilevel_all_gaps_initialization,
            args.bilevel_minimum_class_gap_degrees,
            args.bilevel_gap_prior_weight,
            args.all_gap_learning_source,
            args.hypo_loss_weight,
            args.hypo_compactness_weight,
            args.hypo_temperature,
            args.hypo_prototype_momentum,
        ),
        "seed_{}".format(args.seed),
    )
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.overwrite:
        raise FileExistsError(
            "{} is not empty; use --overwrite to replace matching files".format(
                run_dir
            )
        )
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), vars(args))
    write_json(
        os.path.join(run_dir, "split_ids.json"),
        loaders["split_ids"],
    )

    prior_class_angles = (
        None
        if args.experiment_mode in HYPO_MODES
        else build_iemocap_angles(
            device=device,
            geometry=args.circular_geometry,
            vad_center=(
                args.vad_center_valence,
                args.vad_center_arousal,
            ),
            minimum_confusion_gap_degrees=(
                args.minimum_confusion_gap_degrees
            ),
        )
    )
    model = SDTCSEModel(
        d_text=args.text_dim,
        d_visual=args.visual_dim,
        d_audio=args.audio_dim,
        n_head=args.n_head,
        n_classes=6,
        hidden_dim=args.hidden_dim,
        n_speakers=2,
        dropout=args.dropout,
        experiment_mode=args.experiment_mode,
        embedding_dim=args.embedding_dim,
        projection_dropout=args.projection_dropout,
        initial_cosine_scale=args.initial_cosine_scale,
        initial_class_angles=prior_class_angles,
        bilevel_gap_minimum_degrees=(
            args.bilevel_gap_minimum_degrees
        ),
        bilevel_gap_maximum_degrees=(
            args.bilevel_gap_maximum_degrees
        ),
        bilevel_gap_initial_degrees=(
            args.bilevel_gap_initial_degrees
        ),
        bilevel_minimum_class_gap_degrees=(
            args.bilevel_minimum_class_gap_degrees
        ),
        sdt_residual_update=args.sdt_residual_update,
        spherical_attention_alpha_init=(
            args.spherical_attention_alpha_init
        ),
        spherical_mlp_alpha_init=args.spherical_mlp_alpha_init,
    ).to(device)
    class_weights = iemocap_class_weights(device=device)
    if args.experiment_mode not in HYPO_MODES:
        initial_angle_payload = angle_state_payload(
            model, prior_class_angles
        )
        initial_gap_payload = confusion_gap_payload(
            model,
            prior_class_angles,
            args.minimum_confusion_gap_degrees,
        )
        write_json(
            os.path.join(run_dir, "circular_geometry.json"),
            {
                "geometry": args.circular_geometry,
                "angle_weight": args.angle_weight,
                "confusion_gap_weight": args.confusion_gap_weight,
                "confused_cse_pair_weight": (
                    args.confused_cse_pair_weight
                ),
                "confusion_classification_margin": (
                    args.confusion_classification_margin
                ),
                "confusion_classification_weight": (
                    args.confusion_classification_weight
                ),
                **initial_gap_payload,
                "vad_center": [
                    args.vad_center_valence,
                    args.vad_center_arousal,
                ],
                "nrc_vad_anchors": (
                    build_iemocap_vad_anchors().tolist()
                ),
                **initial_angle_payload,
                "bilevel_geometry": bilevel_gap_payload(
                    model, args
                ),
            },
        )
    circular_loss_function = (
        CircularCSELoss(
            class_angles=prior_class_angles,
            same_class_margin=args.same_class_margin,
            confusion_pair_weight=args.confused_cse_pair_weight,
        ).to(device)
        if args.experiment_mode in CIRCULAR_CSE_MODES
        else None
    )
    hypo_loss_function = (
        HypoPrototypeLoss(
            num_classes=6,
            embedding_dim=args.embedding_dim,
            temperature=args.hypo_temperature,
            prototype_momentum=args.hypo_prototype_momentum,
        ).to(device)
        if args.experiment_mode in HYPO_MODES
        else None
    )
    validation_gap_learning = uses_validation_gap_learning(args)
    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        exclude_angle_parameters=validation_gap_learning,
    )
    angle_optimizer = (
        build_bilevel_angle_optimizer(
            model,
            learning_rate=args.bilevel_angle_learning_rate,
        )
        if validation_gap_learning
        else None
    )

    best_path = os.path.join(run_dir, "best_checkpoint.pt")
    best_f1 = -float("inf")
    best_ce = float("inf")
    best_epoch = None
    best_selection = None
    epoch_rows = []
    angle_history_rows = []
    bilevel_history_rows = []
    residual_gate_history_rows = []
    selection_name = (
        "testing"
        if args.selection_protocol == "test"
        else "validation"
    )
    selection_loader = loaders[selection_name]
    bilevel_validation_batches = (
        cycle_batches(loaders["validation"])
        if validation_gap_learning
        else None
    )

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        training = run_epoch(
            model,
            loaders["training"],
            device,
            class_weights,
            circular_loss_function,
            args,
            optimizer=optimizer,
            bilevel_angle_optimizer=angle_optimizer,
            bilevel_validation_batches=bilevel_validation_batches,
            hypo_loss_function=hypo_loss_function,
        )
        if (
            epoch == 1
            and hypo_loss_function is not None
            and hypo_loss_function.initialized_classes < 6
        ):
            raise RuntimeError(
                "HYPO prototype coverage is incomplete after the "
                "first training epoch; missing class IDs: {}".format(
                    hypo_loss_function.missing_class_ids()
                )
            )
        with torch.no_grad():
            selection = run_epoch(
                model,
                selection_loader,
                device,
                class_weights,
                circular_loss_function,
                args,
                hypo_loss_function=hypo_loss_function,
            )
        row = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "training": public_metrics(training),
            selection_name: public_metrics(selection),
        }
        epoch_rows.append(row)
        write_epoch_metrics(
            os.path.join(run_dir, "epoch_metrics.csv"),
            epoch_rows,
        )
        current_history = (
            None
            if args.experiment_mode in HYPO_MODES
            else angle_history_row(
                epoch,
                model,
                prior_class_angles,
                minimum_confusion_gap_degrees=(
                    args.minimum_confusion_gap_degrees
                    if args.experiment_mode in CONFUSION_GAP_MODES
                    else None
                ),
            )
        )
        if current_history is not None:
            angle_history_rows.append(current_history)
            write_rows(
                os.path.join(run_dir, "angle_history.csv"),
                angle_history_rows,
            )
        if args.experiment_mode in BILEVEL_ANGLE_MODES:
            current_bilevel_geometry = bilevel_gap_payload(
                model, args
            )
            bilevel_row = {
                "epoch": epoch,
                **current_bilevel_geometry,
            }
            for gap_name, gap_value in (
                current_bilevel_geometry.get(
                    "named_gaps_degrees", {}
                ).items()
            ):
                bilevel_row[
                    "gap_{}_degrees".format(gap_name)
                ] = gap_value
            for name in BILEVEL_METRIC_NAMES:
                bilevel_row[name] = training.get(name)
            bilevel_history_rows.append(bilevel_row)
            write_rows(
                os.path.join(
                    run_dir, "bilevel_gap_history.csv"
                ),
                bilevel_history_rows,
            )
        residual_history = spherical_residual_history_row(
            epoch, model
        )
        if residual_history is not None:
            residual_gate_history_rows.append(residual_history)
            write_rows(
                os.path.join(
                    run_dir, "residual_gate_history.csv"
                ),
                residual_gate_history_rows,
            )
        print(
            "epoch={} train_total={:.6f} train_wf1={:.4f} "
            "{}_total={:.6f} {}_wf1={:.4f}".format(
                epoch,
                training["total_loss"],
                training["weighted_f1"],
                selection_name,
                selection["total_loss"],
                selection_name,
                selection["weighted_f1"],
            ),
            flush=True,
        )
        if is_better_selection(
            selection["weighted_f1"],
            selection["fusion_ce"],
            best_f1,
            best_ce,
            args.selection_protocol,
        ):
            best_f1 = selection["weighted_f1"]
            best_ce = selection["fusion_ce"]
            best_epoch = epoch
            best_selection = public_metrics(selection)
            save_checkpoint(
                best_path,
                model,
                optimizer,
                args,
                loaders["split_ids"],
                epoch,
                validation_metrics=(
                    best_selection
                    if args.selection_protocol == "validation"
                    else None
                ),
                selection_metrics=best_selection,
                angle_optimizer=angle_optimizer,
                bilevel_metrics=(
                    {
                        name: training.get(name)
                        for name in BILEVEL_METRIC_NAMES
                    }
                    if validation_gap_learning
                    else None
                ),
                hypo_loss_function=hypo_loss_function,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if hypo_loss_function is not None:
        hypo_state_dict = checkpoint.get("hypo_state_dict")
        if hypo_state_dict is None:
            raise RuntimeError(
                "selected HYPO checkpoint has no prototype state"
            )
        hypo_loss_function.load_state_dict(hypo_state_dict)
    if args.experiment_mode in HYPO_MODES:
        selected_angle_payload = {
            "prior_angles_radians": None,
            "class_angles_radians": None,
            "normalized_gaps_radians": None,
            "angle_offsets_radians": None,
            "angle_regularization": None,
            "target_similarity": None,
        }
        selected_class_angles = None
        selected_gap_payload = {
            "minimum_confusion_gap_degrees": None,
            "confusion_pairs": None,
            "confusion_pair_gaps_degrees": None,
            "confusion_gap_regularization": None,
        }
    else:
        selected_angle_payload = angle_state_payload(
            model, prior_class_angles
        )
        selected_class_angles = current_angle_state(
            model, prior_class_angles
        )["angles"].detach()
        selected_gap_payload = confusion_gap_payload(
            model,
            prior_class_angles,
            args.minimum_confusion_gap_degrees,
        )
    hypo_geometry = save_hypo_artifacts(
        run_dir,
        hypo_loss_function,
        args,
        best_epoch,
    )
    selected_target_similarity = (
        None
        if hypo_loss_function is None
        else hypo_loss_function.prototype_similarity()
    )
    if model.circular_angle_learner is not None:
        write_json(
            os.path.join(
                run_dir, "learned_circular_geometry.json"
            ),
            {
                "selected_epoch": best_epoch,
                "geometry": args.circular_geometry,
                "angle_weight": args.angle_weight,
                "confusion_gap_weight": args.confusion_gap_weight,
                **selected_gap_payload,
                "vad_center": [
                    args.vad_center_valence,
                    args.vad_center_arousal,
                ],
                "nrc_vad_anchors": (
                    build_iemocap_vad_anchors().tolist()
                ),
                **selected_angle_payload,
                "bilevel_geometry": bilevel_gap_payload(
                    model, args
                ),
            },
        )
    with torch.no_grad():
        (
            selected_training,
            training_residual_diagnostics,
        ) = collect_split_with_residual_diagnostics(
            model,
            loaders["training_export"],
            device,
            class_weights,
            circular_loss_function,
            args,
            hypo_loss_function=hypo_loss_function,
        )
        if loaders["validation"] is not None:
            (
                selected_validation,
                validation_residual_diagnostics,
            ) = collect_split_with_residual_diagnostics(
                model,
                loaders["validation"],
                device,
                class_weights,
                circular_loss_function,
                args,
                hypo_loss_function=hypo_loss_function,
            )
        else:
            selected_validation = None
            validation_residual_diagnostics = None
        (
            testing,
            testing_residual_diagnostics,
        ) = collect_split_with_residual_diagnostics(
            model,
            loaders["testing"],
            device,
            class_weights,
            circular_loss_function,
            args,
            hypo_loss_function=hypo_loss_function,
        )

    residual_diagnostics = None
    if args.sdt_residual_update == "spherical":
        residual_diagnostics = {
            "selected_epoch": best_epoch,
            "sdt_residual_update": args.sdt_residual_update,
            "spherical_attention_alpha_init": (
                args.spherical_attention_alpha_init
            ),
            "spherical_mlp_alpha_init": (
                args.spherical_mlp_alpha_init
            ),
            "gates": spherical_residual_gate_payload(model),
            "splits": {
                "training": {
                    "available": True,
                    "updates": training_residual_diagnostics,
                },
                "validation": {
                    "available": (
                        validation_residual_diagnostics is not None
                    ),
                    "updates": validation_residual_diagnostics,
                },
                "testing": {
                    "available": True,
                    "updates": testing_residual_diagnostics,
                },
            },
        }
        write_json(
            os.path.join(
                run_dir, "spherical_residual_diagnostics.json"
            ),
            residual_diagnostics,
        )

    training_metrics = public_metrics(selected_training)
    validation_metrics = (
        public_metrics(selected_validation)
        if selected_validation is not None
        else None
    )
    validation_report = {
        "selected_epoch": best_epoch,
        "available": selected_validation is not None,
    }
    if validation_metrics is None:
        validation_report["reason"] = (
            "selection_protocol=test uses all trainVid for training "
            "and has no validation split"
        )
    else:
        validation_report.update(validation_metrics)
    write_json(
        os.path.join(run_dir, "validation_metrics.json"),
        validation_report,
    )
    test_metrics = public_metrics(testing)
    test_metrics.update(
        final_classification_details(
            testing["labels_array"],
            testing["predictions_array"],
        )
    )
    test_metrics["selected_epoch"] = best_epoch
    write_json(
        os.path.join(run_dir, "test_metrics.json"),
        test_metrics,
    )
    write_rows(
        os.path.join(run_dir, "test_predictions.csv"),
        testing["prediction_rows"],
    )
    save_feature_npz(
        os.path.join(run_dir, "features_train.npz"),
        selected_training,
    )
    if selected_validation is None:
        save_empty_feature_npz(
            os.path.join(run_dir, "features_valid.npz"),
            model,
        )
    else:
        save_feature_npz(
            os.path.join(run_dir, "features_valid.npz"),
            selected_validation,
        )
    save_feature_npz(
        os.path.join(run_dir, "features_test.npz"),
        testing,
    )

    archive = {
        "labels": testing["labels_array"],
        "predictions": testing["predictions_array"],
        "fusion_features": testing["fusion_features_array"],
    }
    if testing["embeddings_array"] is not None:
        archive["embeddings"] = testing["embeddings_array"]
    for archive_name, result_name in (
        ("text_embeddings", "text_embeddings_array"),
        ("audio_embeddings", "audio_embeddings_array"),
        ("visual_embeddings", "visual_embeddings_array"),
    ):
        if testing[result_name] is not None:
            archive[archive_name] = testing[result_name]
    np.savez_compressed(
        os.path.join(run_dir, "test_representations.npz"),
        **archive
    )

    geometry_dir = os.path.join(run_dir, "geometry")
    geometry_target_arguments = (
        {"target_similarity": selected_target_similarity}
        if selected_target_similarity is not None
        else {"class_angles": selected_class_angles}
    )
    fusion_geometry = save_geometry_artifacts(
        geometry_dir,
        "fusion_features",
        testing["fusion_features_array"],
        testing["labels_array"],
        **geometry_target_arguments
    )
    projected_geometry = None
    if testing["embeddings_array"] is not None:
        projected_geometry = save_geometry_artifacts(
            geometry_dir,
            "embeddings",
            testing["embeddings_array"],
            testing["labels_array"],
            **geometry_target_arguments
        )
    unimodal_projected_geometry = {}
    for representation_name, result_name in (
        ("text_embeddings", "text_embeddings_array"),
        ("audio_embeddings", "audio_embeddings_array"),
        ("visual_embeddings", "visual_embeddings_array"),
    ):
        if testing[result_name] is not None:
            unimodal_projected_geometry[representation_name] = (
                save_geometry_artifacts(
                    geometry_dir,
                    representation_name,
                    testing[result_name],
                    testing["labels_array"],
                    **geometry_target_arguments
                )
            )
    summary = {
        "experiment_mode": args.experiment_mode,
        "sdt_residual_update": args.sdt_residual_update,
        "spherical_attention_alpha_init": (
            args.spherical_attention_alpha_init
        ),
        "spherical_mlp_alpha_init": args.spherical_mlp_alpha_init,
        "spherical_residual_gates": (
            spherical_residual_gate_payload(model)
        ),
        "spherical_residual_diagnostics_path": (
            "spherical_residual_diagnostics.json"
            if args.sdt_residual_update == "spherical"
            else None
        ),
        "selection_protocol": args.selection_protocol,
        "selection_split": selection_name,
        "selection_metrics": best_selection,
        "seed": args.seed,
        "hypo_loss_weight": args.hypo_loss_weight,
        "hypo_compactness_weight": (
            args.hypo_compactness_weight
        ),
        "hypo_temperature": args.hypo_temperature,
        "hypo_prototype_momentum": (
            args.hypo_prototype_momentum
        ),
        "hypo_initialized_classes": (
            None
            if hypo_loss_function is None
            else hypo_loss_function.initialized_classes
        ),
        "hypo_prototype_coverage": (
            None
            if hypo_loss_function is None
            else hypo_loss_function.prototype_coverage
        ),
        "hypo_geometry": hypo_geometry,
        "hypo_prototypes_path": (
            "hypo_prototypes.npz"
            if hypo_loss_function is not None
            else None
        ),
        "hypo_geometry_path": (
            "hypo_geometry.json"
            if hypo_loss_function is not None
            else None
        ),
        "circular_weight": args.circular_weight,
        "all_gap_learning_source": (
            args.all_gap_learning_source
        ),
        "angle_weight": args.angle_weight,
        "confusion_gap_weight": args.confusion_gap_weight,
        "confused_cse_pair_weight": args.confused_cse_pair_weight,
        "confusion_classification_margin": (
            args.confusion_classification_margin
        ),
        "confusion_classification_weight": (
            args.confusion_classification_weight
        ),
        **selected_gap_payload,
        "circular_geometry": (
            None
            if args.experiment_mode in HYPO_MODES
            else args.circular_geometry
        ),
        "vad_center": (
            None
            if args.experiment_mode in HYPO_MODES
            else [
                args.vad_center_valence,
                args.vad_center_arousal,
            ]
        ),
        "prior_class_angles": selected_angle_payload[
            "prior_angles_radians"
        ],
        "class_angles": selected_angle_payload[
            "class_angles_radians"
        ],
        "angle_gaps": selected_angle_payload[
            "normalized_gaps_radians"
        ],
        "angle_offsets": selected_angle_payload[
            "angle_offsets_radians"
        ],
        "angle_regularization": selected_angle_payload[
            "angle_regularization"
        ],
        "target_similarity": selected_angle_payload[
            "target_similarity"
        ],
        "bilevel_geometry": bilevel_gap_payload(model, args),
        "selected_epoch_bilevel_metrics": checkpoint.get(
            "bilevel_metrics"
        ),
        "selected_epoch": best_epoch,
        "training": training_metrics,
        "validation": validation_metrics,
        "test": public_metrics(testing),
        "fusion_geometry": fusion_geometry,
        "projected_geometry": projected_geometry,
        "unimodal_projected_geometry": unimodal_projected_geometry,
        "run_directory": run_dir,
    }
    write_json(os.path.join(run_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train SDT and cosine/CircularCSE variants with a fixed "
            "train/validation/test split."
        )
    )
    parser.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default="sdt_cse",
    )
    parser.add_argument("--feature-path", default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--sdt-residual-update",
        choices=SDT_RESIDUAL_UPDATES,
        default="standard",
        help=(
            "standard preserves original SDT residual additions; "
            "spherical uses learned normalized interpolation"
        ),
    )
    parser.add_argument(
        "--spherical-attention-alpha-init",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--spherical-mlp-alpha-init",
        type=float,
        default=0.1,
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument(
        "--initial-cosine-scale", type=float, default=16.0
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--fusion-ce-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--unimodal-ce-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--distillation-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--circular-weight", type=float, default=0.1
    )
    parser.add_argument(
        "--hypo-loss-weight",
        type=float,
        default=0.1,
        help="outer HYPO auxiliary-loss weight",
    )
    parser.add_argument(
        "--hypo-compactness-weight",
        type=float,
        default=2.0,
        help="compactness weight inside the HYPO objective",
    )
    parser.add_argument(
        "--hypo-temperature",
        type=float,
        default=0.1,
        help="temperature for HYPO compactness and dispersion",
    )
    parser.add_argument(
        "--hypo-prototype-momentum",
        type=float,
        default=0.95,
        help="EMA momentum for the training-only HYPO prototypes",
    )
    parser.add_argument(
        "--angle-weight",
        type=float,
        default=0.1,
        help=(
            "prior regularization weight for learnable circular "
            "angles (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--confusion-gap-weight",
        type=float,
        default=0.1,
        help=(
            "weight for the minimum happy-excited and "
            "angry-frustrated angular-gap penalty in confusion-gap "
            "mode (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--minimum-confusion-gap-degrees",
        type=float,
        default=75.0,
        help=(
            "minimum shortest angle for predefined confusion pairs "
            "in confusion-gap mode (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--confused-cse-pair-weight",
        type=float,
        default=5.0,
        help=(
            "relative CircularCSE weight for happy-excited and "
            "angry-frustrated ordered pairs in confusion-margin "
            "mode (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--confusion-classification-margin",
        type=float,
        default=0.1,
        help=(
            "required true-minus-confused raw cosine-score margin "
            "in confusion-margin mode (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--confusion-classification-weight",
        type=float,
        default=0.1,
        help=(
            "weight for the direct confusion-aware cosine "
            "classification margin (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--bilevel-gap-minimum-degrees",
        type=float,
        default=70.0,
        help="lower bound for the validation-learned shared gap",
    )
    parser.add_argument(
        "--bilevel-gap-maximum-degrees",
        type=float,
        default=110.0,
        help="upper bound for the validation-learned shared gap",
    )
    parser.add_argument(
        "--bilevel-gap-initial-degrees",
        type=float,
        default=90.0,
        help="initial validation-learned shared gap",
    )
    parser.add_argument(
        "--bilevel-all-gaps-initialization",
        choices=("equal", "nrc_vad"),
        default="equal",
        help=(
            "fixed geometry used to initialize the six independently "
            "learned hard-floor gaps"
        ),
    )
    parser.add_argument(
        "--all-gap-learning-source",
        choices=("validation", "training"),
        default="validation",
        help=(
            "validation uses bilevel hypergradients; training puts "
            "the same six hard-floor gap parameters in the ordinary "
            "training optimizer"
        ),
    )
    parser.add_argument(
        "--bilevel-minimum-class-gap-degrees",
        type=float,
        default=20.0,
        help=(
            "hard lower bound for every independently learned "
            "consecutive class gap"
        ),
    )
    parser.add_argument(
        "--bilevel-gap-prior-weight",
        type=float,
        default=0.01,
        help=(
            "outer-objective weight for squared deviation from the "
            "six initialization gaps"
        ),
    )
    parser.add_argument(
        "--bilevel-angle-learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate for the outer gap parameter",
    )
    parser.add_argument(
        "--bilevel-inner-step-size",
        type=float,
        default=None,
        help=(
            "one-step inner SGD size used by the hypergradient; "
            "defaults to --lr"
        ),
    )
    parser.add_argument(
        "--bilevel-hvp-radius",
        type=float,
        default=0.01,
        help="finite-difference radius for the DARTS Hessian-vector product",
    )
    parser.add_argument(
        "--bilevel-outer-confusion-weight",
        type=float,
        default=0.1,
        help=(
            "confusion-margin weight in the validation outer "
            "classification objective"
        ),
    )
    parser.add_argument(
        "--bilevel-angle-gradient-clip",
        type=float,
        default=5.0,
        help=(
            "elementwise absolute clipping bound for gap "
            "hypergradients; 0 disables"
        ),
    )
    parser.add_argument(
        "--same-class-margin", type=float, default=0.0
    )
    parser.add_argument(
        "--circular-geometry",
        choices=CIRCULAR_GEOMETRIES,
        default=None,
        help=(
            "equal uses the original six equally spaced angles; "
            "nrc_vad derives nonuniform angles from NRC-VAD anchors; "
            "confusion_separated fixes happy-excited and "
            "angry-frustrated at the requested minimum gap and "
            "balances the other four gaps. "
            "Defaults to nrc_vad for sdt_cse_learnable_angles and "
            "confusion_separated for confusion-margin mode"
        ),
    )
    parser.add_argument(
        "--vad-center-valence", type=float, default=0.5
    )
    parser.add_argument(
        "--vad-center-arousal", type=float, default=0.5
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument(
        "--selection-protocol",
        choices=SELECTION_PROTOCOLS,
        default="validation",
        help=(
            "validation selects on the first validation-ratio of "
            "trainVid; test reproduces original SDT by training on "
            "all trainVid and selecting on test weighted F1 every epoch"
        ),
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--text-dim", type=int, default=1024)
    parser.add_argument("--visual-dim", type=int, default=342)
    parser.add_argument("--audio-dim", type=int, default=1582)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "zero-based CUDA device index among visible GPUs "
            "(default: 0)"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_argument_parser().parse_args()
    try:
        train_and_test(args)
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
