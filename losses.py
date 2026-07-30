import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ID2EMOTION = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
    5: "frustrated",
}
IEMOCAP_CONFUSION_PAIRS = (
    (0, 4),  # happy -> excited
    (3, 5),  # angry -> frustrated
)
IEMOCAP_THREE_CONFUSION_GAP_PAIRS = (
    *IEMOCAP_CONFUSION_PAIRS,
    (1, 2),  # sad -> neutral
)
EMOTION_NAMES = [ID2EMOTION[index] for index in range(len(ID2EMOTION))]
IEMOCAP_CLASS_FREQUENCIES = (
    0.086747,
    0.144406,
    0.227883,
    0.160585,
    0.127711,
    0.252668,
)
CIRCULAR_GEOMETRIES = (
    "equal",
    "nrc_vad",
    "confusion_separated",
)
NRC_VAD_ANCHORS = (
    (0.960, 0.732),  # happy
    (0.052, 0.288),  # sad
    (0.469, 0.184),  # neutral
    (0.167, 0.865),  # angry
    (0.908, 0.931),  # excited
    (0.060, 0.730),  # frustrated
)


def build_iemocap_vad_anchors(device=None, dtype=torch.float32):
    return torch.tensor(
        NRC_VAD_ANCHORS,
        device=device,
        dtype=dtype,
    )


def build_iemocap_angles(
    device=None,
    dtype=torch.float32,
    geometry="equal",
    vad_center=(0.5, 0.5),
    minimum_confusion_gap_degrees=75.0,
):
    if geometry not in CIRCULAR_GEOMETRIES:
        raise ValueError(
            "geometry must be one of {}".format(CIRCULAR_GEOMETRIES)
        )
    if geometry == "equal":
        return torch.tensor(
            [
                0.0,
                4.0 * math.pi / 3.0,
                5.0 * math.pi / 3.0,
                2.0 * math.pi / 3.0,
                math.pi / 3.0,
                math.pi,
            ],
            device=device,
            dtype=dtype,
        )
    if geometry == "confusion_separated":
        minimum_gap = float(minimum_confusion_gap_degrees)
        if (
            not math.isfinite(minimum_gap)
            or minimum_gap <= 0.0
            or minimum_gap >= 180.0
        ):
            raise ValueError(
                "minimum_confusion_gap_degrees must be finite and "
                "in (0, 180) for confusion-separated geometry"
            )
        remaining_gap = (360.0 - 2.0 * minimum_gap) / 4.0
        if remaining_gap <= 0.0:
            raise ValueError(
                "confusion-separated geometry requires positive "
                "non-confusion gaps"
            )
        # Circular order: happy, excited, angry, frustrated, sad,
        # neutral. The two predefined confusion gaps are fixed to the
        # requested value and the remaining circumference is balanced.
        ordered_degrees = torch.tensor(
            [
                0.0,
                minimum_gap,
                minimum_gap + remaining_gap,
                2.0 * minimum_gap + remaining_gap,
                2.0 * minimum_gap + 2.0 * remaining_gap,
                2.0 * minimum_gap + 3.0 * remaining_gap,
            ],
            device=device,
            dtype=dtype,
        )
        circle_order = torch.tensor(
            [0, 4, 3, 5, 1, 2],
            device=device,
            dtype=torch.long,
        )
        angles = torch.empty(
            6,
            device=device,
            dtype=dtype,
        )
        angles[circle_order] = torch.deg2rad(ordered_degrees)
        return angles

    center = torch.as_tensor(
        vad_center,
        device=device,
        dtype=dtype,
    )
    if center.shape != (2,):
        raise ValueError("vad_center must contain valence and arousal")
    if not torch.isfinite(center).all():
        raise ValueError("vad_center must contain only finite values")
    anchors = build_iemocap_vad_anchors(device=device, dtype=dtype)
    offsets = anchors - center
    if torch.any(torch.linalg.vector_norm(offsets, dim=-1) <= 1e-8):
        raise ValueError(
            "a VAD anchor coincides with the center and has no angle"
        )
    angles = torch.atan2(offsets[:, 1], offsets[:, 0])
    return torch.remainder(
        angles,
        angles.new_tensor(2.0 * math.pi),
    )


def build_target_similarity(class_angles):
    if class_angles.ndim != 1:
        raise ValueError("class_angles must be one-dimensional")
    return torch.cos(
        class_angles[:, None] - class_angles[None, :]
    )


def circular_pair_distances(
    class_angles,
    pairs=IEMOCAP_CONFUSION_PAIRS,
):
    """Return shortest circular distances for selected class pairs."""
    if class_angles.ndim != 1:
        raise ValueError("class_angles must be one-dimensional")
    if not torch.isfinite(class_angles).all():
        raise ValueError(
            "class_angles must contain only finite values"
        )
    if not pairs:
        raise ValueError("pairs must not be empty")
    pair_ids = torch.as_tensor(
        pairs,
        dtype=torch.long,
        device=class_angles.device,
    )
    if pair_ids.ndim != 2 or pair_ids.size(1) != 2:
        raise ValueError("pairs must have shape [P, 2]")
    if pair_ids.min().item() < 0:
        raise ValueError("pair class IDs must be nonnegative")
    if pair_ids.max().item() >= class_angles.numel():
        raise ValueError("pair class ID exceeds configured angles")
    differences = (
        class_angles[pair_ids[:, 0]]
        - class_angles[pair_ids[:, 1]]
    )
    two_pi = differences.new_tensor(2.0 * math.pi)
    wrapped = torch.remainder(torch.abs(differences), two_pi)
    return torch.minimum(
        wrapped,
        two_pi - wrapped,
    )


def minimum_confusion_gap_regularization(
    class_angles,
    minimum_gap_degrees=75.0,
    pairs=IEMOCAP_CONFUSION_PAIRS,
):
    """Penalize predefined confusion-pair angles below a minimum."""
    minimum_gap_degrees = float(minimum_gap_degrees)
    if (
        not math.isfinite(minimum_gap_degrees)
        or minimum_gap_degrees <= 0.0
        or minimum_gap_degrees > 180.0
    ):
        raise ValueError(
            "minimum_gap_degrees must be finite and in (0, 180]"
        )
    distances = circular_pair_distances(class_angles, pairs)
    minimum = distances.new_tensor(
        math.radians(minimum_gap_degrees)
    )
    return F.relu(minimum - distances).pow(2).sum()


def iemocap_class_weights(device=None, dtype=torch.float32):
    return torch.tensor(
        [1.0 / frequency for frequency in IEMOCAP_CLASS_FREQUENCIES],
        device=device,
        dtype=dtype,
    )


def _valid_flattened(logits, labels, mask):
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, L, C]")
    valid = mask.reshape(-1) > 0
    return (
        logits.reshape(-1, logits.size(-1))[valid],
        labels.reshape(-1)[valid],
        valid,
    )


def masked_weighted_cross_entropy(logits, labels, mask, class_weights):
    valid_logits, valid_labels, _ = _valid_flattened(
        logits, labels, mask
    )
    if valid_labels.numel() == 0:
        return logits.sum() * 0.0
    summed = F.cross_entropy(
        valid_logits,
        valid_labels,
        weight=class_weights,
        reduction="sum",
    )
    denominator = class_weights[valid_labels].sum()
    return summed / denominator


def masked_self_distillation_kl(
    student_logits,
    teacher_logits,
    mask,
    temperature=1.0,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = mask.reshape(-1) > 0
    if valid.sum().item() == 0:
        return student_logits.sum() * 0.0
    student = F.log_softmax(
        student_logits / temperature, dim=-1
    ).reshape(-1, student_logits.size(-1))[valid]
    teacher = F.softmax(
        teacher_logits / temperature, dim=-1
    ).reshape(-1, teacher_logits.size(-1))[valid]
    return F.kl_div(student, teacher, reduction="sum") / valid.sum().float()


def masked_confusion_classification_margin(
    cosine_scores,
    labels,
    mask,
    margin=0.1,
    pairs=IEMOCAP_CONFUSION_PAIRS,
):
    """Require the true cosine score to exceed a confused competitor."""
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0 or margin > 2.0:
        raise ValueError("margin must be finite and in [0, 2]")
    valid_scores, valid_labels, _ = _valid_flattened(
        cosine_scores,
        labels,
        mask,
    )
    if not pairs:
        raise ValueError("pairs must not be empty")
    competitor_by_class = {}
    for first_id, second_id in pairs:
        first_id = int(first_id)
        second_id = int(second_id)
        if first_id == second_id:
            raise ValueError(
                "confusion pairs must contain different classes"
            )
        if (
            first_id in competitor_by_class
            or second_id in competitor_by_class
        ):
            raise ValueError(
                "each class may occur in at most one confusion pair"
            )
        competitor_by_class[first_id] = second_id
        competitor_by_class[second_id] = first_id
    if competitor_by_class:
        minimum_id = min(competitor_by_class)
        maximum_id = max(competitor_by_class)
        if minimum_id < 0 or maximum_id >= cosine_scores.size(-1):
            raise ValueError(
                "confusion pair class ID exceeds cosine scores"
            )

    selected = torch.zeros_like(valid_labels, dtype=torch.bool)
    competitors = torch.zeros_like(valid_labels)
    for class_id, competitor_id in competitor_by_class.items():
        class_selected = valid_labels.eq(class_id)
        selected = selected | class_selected
        competitors[class_selected] = competitor_id
    if selected.sum().item() == 0:
        return cosine_scores.sum() * 0.0

    selected_scores = valid_scores[selected]
    selected_labels = valid_labels[selected]
    selected_competitors = competitors[selected]
    row_ids = torch.arange(
        selected_scores.size(0),
        device=selected_scores.device,
    )
    true_scores = selected_scores[row_ids, selected_labels]
    competing_scores = selected_scores[
        row_ids, selected_competitors
    ]
    return F.relu(
        true_scores.new_tensor(margin)
        - (true_scores - competing_scores)
    ).mean()


class CircularCSELoss(nn.Module):
    def __init__(
        self,
        class_angles=None,
        same_class_margin=0.0,
        confusion_pair_weight=1.0,
        confusion_pairs=IEMOCAP_CONFUSION_PAIRS,
    ):
        super().__init__()
        if same_class_margin < 0:
            raise ValueError("same_class_margin must be nonnegative")
        confusion_pair_weight = float(confusion_pair_weight)
        if (
            not math.isfinite(confusion_pair_weight)
            or confusion_pair_weight < 1.0
        ):
            raise ValueError(
                "confusion_pair_weight must be finite and at least 1"
            )
        if not confusion_pairs:
            raise ValueError("confusion_pairs must not be empty")
        if class_angles is None:
            class_angles = build_iemocap_angles()
        if class_angles.ndim != 1:
            raise ValueError("class_angles must be one-dimensional")
        self.same_class_margin = float(same_class_margin)
        self.confusion_pair_weight = confusion_pair_weight
        self.confusion_pairs = tuple(
            (int(first_id), int(second_id))
            for first_id, second_id in confusion_pairs
        )
        self.register_buffer(
            "class_angles", class_angles.detach().clone()
        )
        self.register_buffer(
            "target_similarity",
            build_target_similarity(class_angles).detach().clone(),
        )

    def forward(self, embeddings, labels, class_angles=None):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [N, D]")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [N]")
        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                "embeddings and labels must share their first dimension"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        active_class_angles = (
            self.class_angles
            if class_angles is None
            else class_angles
        )
        if active_class_angles.ndim != 1:
            raise ValueError("class_angles must be one-dimensional")
        if not torch.isfinite(active_class_angles).all():
            raise ValueError(
                "class_angles must contain only finite values"
            )
        if labels.numel() > 0:
            if labels.min().item() < 0:
                raise ValueError("labels must be nonnegative")
            if labels.max().item() >= active_class_angles.numel():
                raise ValueError("label exceeds the configured class count")

        embeddings = F.normalize(
            embeddings, p=2, dim=-1, eps=1e-8
        )
        count = embeddings.size(0)
        if count < 2:
            return embeddings.sum() * 0.0

        predicted = torch.matmul(embeddings, embeddings.t())
        if class_angles is None:
            target_table = self.target_similarity.to(
                device=predicted.device,
                dtype=predicted.dtype,
            )
        else:
            dynamic_angles = active_class_angles.to(
                device=predicted.device,
                dtype=predicted.dtype,
            )
            target_table = build_target_similarity(dynamic_angles)
        target = target_table[labels[:, None], labels[None, :]]
        same_class = labels[:, None].eq(labels[None, :])
        off_diagonal = ~torch.eye(
            count, dtype=torch.bool, device=embeddings.device
        )

        same_error = torch.abs(predicted - target)
        same_loss = F.relu(
            same_error - self.same_class_margin
        ).pow(2)
        different_loss = (predicted - target).pow(2)
        pairwise = torch.where(
            same_class, same_loss, different_loss
        )
        if self.confusion_pair_weight == 1.0:
            return pairwise[off_diagonal].mean()

        pair_weights = torch.ones_like(pairwise)
        for first_id, second_id in self.confusion_pairs:
            selected_pair = (
                labels[:, None].eq(first_id)
                & labels[None, :].eq(second_id)
            ) | (
                labels[:, None].eq(second_id)
                & labels[None, :].eq(first_id)
            )
            pair_weights = torch.where(
                selected_pair,
                pair_weights.new_tensor(
                    self.confusion_pair_weight
                ),
                pair_weights,
            )
        valid_losses = pairwise[off_diagonal]
        valid_weights = pair_weights[off_diagonal]
        return (
            valid_losses * valid_weights
        ).sum() / valid_weights.sum()


class HypoPrototypeLoss(nn.Module):
    """Prior-free HYPO compactness and dispersion with EMA prototypes."""

    def __init__(
        self,
        num_classes,
        embedding_dim,
        temperature=0.1,
        prototype_momentum=0.95,
    ):
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least 2")
        if int(embedding_dim) < 1:
            raise ValueError("embedding_dim must be positive")
        temperature = float(temperature)
        prototype_momentum = float(prototype_momentum)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if (
            not math.isfinite(prototype_momentum)
            or prototype_momentum < 0.0
            or prototype_momentum >= 1.0
        ):
            raise ValueError(
                "prototype_momentum must be finite and in [0, 1)"
            )

        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.temperature = temperature
        self.prototype_momentum = prototype_momentum
        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.embedding_dim),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )
        self.register_buffer(
            "update_counts",
            torch.zeros(self.num_classes, dtype=torch.long),
        )

    @property
    def initialized_classes(self):
        return int(self.initialized.sum().item())

    @property
    def prototype_coverage(self):
        return self.initialized_classes / float(self.num_classes)

    def missing_class_ids(self):
        return (
            torch.nonzero(~self.initialized, as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

    def prototype_similarity(self):
        if not bool(self.initialized.all().item()):
            return None
        prototypes = F.normalize(
            self.prototypes, p=2, dim=-1, eps=1e-8
        )
        return torch.matmul(prototypes, prototypes.t())

    def forward(
        self,
        embeddings,
        labels,
        update_prototypes=False,
        use_batch_candidates=None,
        target_similarity=None,
    ):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [N, D]")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                "embedding dimension does not match the prototype bank"
            )
        if labels.ndim != 1:
            raise ValueError("labels must have shape [N]")
        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                "embeddings and labels must share their first dimension"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        if labels.numel() > 0:
            if labels.min().item() < 0:
                raise ValueError("labels must be nonnegative")
            if labels.max().item() >= self.num_classes:
                raise ValueError("label exceeds the configured class count")
        if use_batch_candidates is None:
            use_batch_candidates = bool(update_prototypes)
        else:
            use_batch_candidates = bool(use_batch_candidates)
        if update_prototypes and not use_batch_candidates:
            raise ValueError(
                "prototype updates require batch candidates"
            )

        embeddings = F.normalize(
            embeddings, p=2, dim=-1, eps=1e-8
        )
        zero = embeddings.sum() * 0.0
        if embeddings.size(0) == 0:
            return {
                "compactness": zero,
                "dispersion": zero,
                "alignment": zero,
                "active": False,
            }

        observed = []
        candidate_prototypes = []
        for class_id in range(self.num_classes):
            class_selected = labels.eq(class_id)
            class_observed = bool(class_selected.any().item())
            observed.append(class_observed)
            current = self.prototypes[class_id].detach()
            if not use_batch_candidates or not class_observed:
                candidate = current
            else:
                class_mean = F.normalize(
                    embeddings[class_selected].mean(dim=0),
                    p=2,
                    dim=0,
                    eps=1e-8,
                )
                if bool(self.initialized[class_id].item()):
                    candidate = F.normalize(
                        self.prototype_momentum * current
                        + (1.0 - self.prototype_momentum) * class_mean,
                        p=2,
                        dim=0,
                        eps=1e-8,
                    )
                else:
                    candidate = class_mean
                candidate_norm = torch.linalg.vector_norm(candidate)
                if (
                    not bool(torch.isfinite(candidate_norm).item())
                    or abs(float(candidate_norm.detach().item()) - 1.0)
                    > 1e-4
                ):
                    raise RuntimeError(
                        "cannot form a unit HYPO prototype for class "
                        "{} from a degenerate class mean".format(
                            class_id
                        )
                    )
            candidate_prototypes.append(candidate)
        candidates = torch.stack(candidate_prototypes, dim=0)

        if update_prototypes:
            observed_mask = torch.tensor(
                observed,
                device=self.initialized.device,
                dtype=torch.bool,
            )
            with torch.no_grad():
                observed_ids = torch.nonzero(
                    observed_mask, as_tuple=False
                ).flatten()
                self.prototypes.index_copy_(
                    0,
                    observed_ids,
                    candidates.index_select(
                        0, observed_ids
                    ).detach(),
                )
                self.initialized.logical_or_(observed_mask)
                self.update_counts.add_(observed_mask.long())

        if not bool(self.initialized.all().item()):
            return {
                "compactness": zero,
                "dispersion": zero,
                "alignment": zero,
                "active": False,
            }

        committed = F.normalize(
            self.prototypes.detach(), p=2, dim=-1, eps=1e-8
        )
        compactness_logits = torch.matmul(
            embeddings, committed.t()
        ) / self.temperature
        compactness = F.cross_entropy(
            compactness_logits, labels, reduction="mean"
        )

        dispersion_prototypes = F.normalize(
            candidates, p=2, dim=-1, eps=1e-8
        )
        prototype_similarity = torch.matmul(
            dispersion_prototypes, dispersion_prototypes.t()
        )
        similarities = prototype_similarity / self.temperature
        off_diagonal = ~torch.eye(
            self.num_classes,
            dtype=torch.bool,
            device=similarities.device,
        )
        negative_similarities = similarities[off_diagonal].reshape(
            self.num_classes, self.num_classes - 1
        )
        dispersion = (
            torch.logsumexp(negative_similarities, dim=1)
            - math.log(self.num_classes - 1)
        ).mean()
        alignment = zero
        if target_similarity is not None:
            if torch.is_tensor(target_similarity):
                target_similarity = target_similarity.to(
                    device=prototype_similarity.device,
                    dtype=prototype_similarity.dtype,
                )
            else:
                target_similarity = torch.as_tensor(
                    target_similarity,
                    device=prototype_similarity.device,
                    dtype=prototype_similarity.dtype,
                )
            if target_similarity.shape != (
                self.num_classes,
                self.num_classes,
            ):
                raise ValueError(
                    "target_similarity must have shape [C, C]"
                )
            if not torch.isfinite(target_similarity).all():
                raise ValueError(
                    "target_similarity must contain finite values"
                )
            alignment = F.mse_loss(
                prototype_similarity[off_diagonal],
                target_similarity[off_diagonal],
                reduction="mean",
            )
        return {
            "compactness": compactness,
            "dispersion": dispersion,
            "alignment": alignment,
            "active": True,
        }


def compute_sdt_cse_losses(
    outputs,
    labels,
    utterance_mask,
    class_weights,
    circular_loss_function=None,
    temperature=1.0,
    fusion_ce_weight=1.0,
    unimodal_ce_weight=1.0,
    distillation_weight=1.0,
    circular_weight=0.0,
    angle_weight=0.0,
    confusion_gap_weight=0.0,
    minimum_confusion_gap_degrees=75.0,
    confusion_gap_pairs=IEMOCAP_CONFUSION_PAIRS,
    confusion_classification_weight=0.0,
    confusion_classification_margin=0.1,
    hypo_loss_function=None,
    hypo_loss_weight=0.0,
    hypo_compactness_weight=2.0,
    update_hypo_prototypes=False,
    use_batch_hypo_candidates=None,
    hypo_alignment_enabled=False,
    hypo_alignment_weight=0.0,
    detach_hypo_alignment_target=False,
):
    for name, value in {
        "fusion_ce_weight": fusion_ce_weight,
        "unimodal_ce_weight": unimodal_ce_weight,
        "distillation_weight": distillation_weight,
        "circular_weight": circular_weight,
        "hypo_loss_weight": hypo_loss_weight,
        "hypo_compactness_weight": hypo_compactness_weight,
        "hypo_alignment_weight": hypo_alignment_weight,
        "angle_weight": angle_weight,
        "confusion_gap_weight": confusion_gap_weight,
        "confusion_classification_weight": (
            confusion_classification_weight
        ),
    }.items():
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(
                "{} must be finite and nonnegative".format(name)
            )

    fusion_ce = masked_weighted_cross_entropy(
        outputs["fusion_logits"],
        labels,
        utterance_mask,
        class_weights,
    )
    unimodal_logits = (
        outputs["text_logits"],
        outputs["audio_logits"],
        outputs["visual_logits"],
    )
    available = tuple(logits is not None for logits in unimodal_logits)
    if any(available) and not all(available):
        raise ValueError(
            "unimodal logits must either all be present or all be None"
        )
    if all(available):
        text_ce = masked_weighted_cross_entropy(
            outputs["text_logits"],
            labels,
            utterance_mask,
            class_weights,
        )
        audio_ce = masked_weighted_cross_entropy(
            outputs["audio_logits"],
            labels,
            utterance_mask,
            class_weights,
        )
        visual_ce = masked_weighted_cross_entropy(
            outputs["visual_logits"],
            labels,
            utterance_mask,
            class_weights,
        )

        text_kl = masked_self_distillation_kl(
            outputs["text_logits"],
            outputs["fusion_logits"],
            utterance_mask,
            temperature,
        )
        audio_kl = masked_self_distillation_kl(
            outputs["audio_logits"],
            outputs["fusion_logits"],
            utterance_mask,
            temperature,
        )
        visual_kl = masked_self_distillation_kl(
            outputs["visual_logits"],
            outputs["fusion_logits"],
            utterance_mask,
            temperature,
        )
    else:
        zero = outputs["fusion_logits"].sum() * 0.0
        text_ce = zero
        audio_ce = zero
        visual_ce = zero
        text_kl = zero
        audio_kl = zero
        visual_kl = zero

    circular_zero = outputs["fusion_logits"].sum() * 0.0
    fusion_circular = circular_zero
    text_circular = circular_zero
    audio_circular = circular_zero
    visual_circular = circular_zero
    if circular_weight > 0:
        if circular_loss_function is None:
            raise ValueError(
                "positive circular_weight requires a CircularCSE loss"
            )
        if outputs["embeddings"] is None:
            raise ValueError(
                "CircularCSE requires projected fusion embeddings"
            )
        valid = utterance_mask.reshape(-1) > 0
        valid_embeddings = outputs["embeddings"].reshape(
            -1, outputs["embeddings"].size(-1)
        )[valid]
        valid_labels = labels.reshape(-1)[valid]
        fusion_circular = circular_loss_function(
            valid_embeddings,
            valid_labels,
            class_angles=outputs.get("class_angles"),
        )
        if outputs.get("unimodal_circular_cse_enabled", False):
            modality_embeddings = (
                outputs.get("text_embeddings"),
                outputs.get("audio_embeddings"),
                outputs.get("visual_embeddings"),
            )
            if not all(
                embeddings is not None
                for embeddings in modality_embeddings
            ):
                raise ValueError(
                    "all-modal CircularCSE requires projected text, "
                    "audio, and visual embeddings"
                )
            modality_losses = []
            for embeddings in modality_embeddings:
                flattened = embeddings.reshape(
                    -1, embeddings.size(-1)
                )[valid]
                modality_losses.append(
                    circular_loss_function(
                        flattened,
                        valid_labels,
                        class_angles=outputs.get("class_angles"),
                    )
                )
            (
                text_circular,
                audio_circular,
                visual_circular,
            ) = modality_losses

    unimodal_ce = text_ce + audio_ce + visual_ce
    distillation = text_kl + audio_kl + visual_kl
    unimodal_circular = (
        text_circular + audio_circular + visual_circular
    )
    total_circular = fusion_circular + unimodal_circular
    hypo_compactness = circular_zero
    hypo_dispersion = circular_zero
    hypo_alignment = circular_zero
    if hypo_loss_function is not None:
        if outputs["embeddings"] is None:
            raise ValueError(
                "HYPO requires projected fusion embeddings"
            )
        valid = utterance_mask.reshape(-1) > 0
        valid_embeddings = outputs["embeddings"].reshape(
            -1, outputs["embeddings"].size(-1)
        )[valid]
        valid_labels = labels.reshape(-1)[valid]
        target_similarity = None
        if hypo_alignment_enabled:
            class_angles = outputs.get("class_angles")
            if class_angles is None:
                raise ValueError(
                    "circle-aligned HYPO requires class angles"
                )
            target_similarity = build_target_similarity(
                class_angles
            )
            if detach_hypo_alignment_target:
                target_similarity = target_similarity.detach()
        hypo_components = hypo_loss_function(
            valid_embeddings,
            valid_labels,
            update_prototypes=update_hypo_prototypes,
            use_batch_candidates=use_batch_hypo_candidates,
            target_similarity=target_similarity,
        )
        hypo_compactness = hypo_components["compactness"]
        if hypo_alignment_enabled:
            hypo_alignment = hypo_components["alignment"]
        else:
            hypo_dispersion = hypo_components["dispersion"]
    elif hypo_loss_weight > 0:
        raise ValueError(
            "positive hypo_loss_weight requires a HYPO prototype loss"
        )
    hypo_total = (
        hypo_compactness_weight * hypo_compactness
        + (
            hypo_alignment_weight * hypo_alignment
            if hypo_alignment_enabled
            else hypo_dispersion
        )
    )
    angle_regularization = outputs["fusion_logits"].sum() * 0.0
    if outputs.get("angle_regularization") is not None:
        angle_regularization = outputs["angle_regularization"]
    elif angle_weight > 0:
        raise ValueError(
            "positive angle_weight requires learnable circular angles"
        )
    confusion_gap_regularization = (
        outputs["fusion_logits"].sum() * 0.0
    )
    if outputs.get("confusion_gap_enabled", False):
        if outputs.get("class_angles") is None:
            raise ValueError(
                "confusion-gap mode requires learnable circular angles"
            )
        confusion_gap_regularization = (
            minimum_confusion_gap_regularization(
                outputs["class_angles"],
                minimum_gap_degrees=(
                    minimum_confusion_gap_degrees
                ),
                pairs=confusion_gap_pairs,
            )
        )
    elif confusion_gap_weight > 0:
        raise ValueError(
            "positive confusion_gap_weight requires confusion-gap mode"
        )
    confusion_classification = (
        outputs["fusion_logits"].sum() * 0.0
    )
    if outputs.get("confusion_margin_enabled", False):
        cosine_scores = outputs.get("fusion_cosine_scores")
        if cosine_scores is None:
            raise ValueError(
                "confusion-margin mode requires fusion cosine scores"
            )
        confusion_classification = (
            masked_confusion_classification_margin(
                cosine_scores,
                labels,
                utterance_mask,
                margin=confusion_classification_margin,
            )
        )
    elif confusion_classification_weight > 0:
        raise ValueError(
            "positive confusion_classification_weight requires "
            "confusion-margin mode"
        )
    total = (
        fusion_ce_weight * fusion_ce
        + unimodal_ce_weight * unimodal_ce
        + distillation_weight * distillation
        + circular_weight * total_circular
        + hypo_loss_weight * hypo_total
        + angle_weight * angle_regularization
        + confusion_gap_weight * confusion_gap_regularization
        + confusion_classification_weight
        * confusion_classification
    )
    return {
        "total_loss": total,
        "fusion_ce": fusion_ce,
        "text_ce": text_ce,
        "audio_ce": audio_ce,
        "visual_ce": visual_ce,
        "unimodal_ce": unimodal_ce,
        "text_kl": text_kl,
        "audio_kl": audio_kl,
        "visual_kl": visual_kl,
        "distillation": distillation,
        "circular_cse": fusion_circular,
        "fusion_circular_cse": fusion_circular,
        "text_circular_cse": text_circular,
        "audio_circular_cse": audio_circular,
        "visual_circular_cse": visual_circular,
        "unimodal_circular_cse": unimodal_circular,
        "total_circular_cse": total_circular,
        "hypo_compactness": hypo_compactness,
        "hypo_dispersion": hypo_dispersion,
        "hypo_alignment": hypo_alignment,
        "hypo_total": hypo_total,
        "angle_regularization": angle_regularization,
        "confusion_gap_regularization": (
            confusion_gap_regularization
        ),
        "confusion_classification_margin": (
            confusion_classification
        ),
    }
