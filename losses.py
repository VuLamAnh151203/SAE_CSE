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
EMOTION_NAMES = [ID2EMOTION[index] for index in range(len(ID2EMOTION))]
IEMOCAP_CLASS_FREQUENCIES = (
    0.086747,
    0.144406,
    0.227883,
    0.160585,
    0.127711,
    0.252668,
)


def build_iemocap_angles(device=None, dtype=torch.float32):
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


def build_target_similarity(class_angles):
    if class_angles.ndim != 1:
        raise ValueError("class_angles must be one-dimensional")
    return torch.cos(
        class_angles[:, None] - class_angles[None, :]
    )


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


class CircularCSELoss(nn.Module):
    def __init__(self, class_angles=None, same_class_margin=0.0):
        super().__init__()
        if same_class_margin < 0:
            raise ValueError("same_class_margin must be nonnegative")
        if class_angles is None:
            class_angles = build_iemocap_angles()
        if class_angles.ndim != 1:
            raise ValueError("class_angles must be one-dimensional")
        self.same_class_margin = float(same_class_margin)
        self.register_buffer(
            "class_angles", class_angles.detach().clone()
        )
        self.register_buffer(
            "target_similarity",
            build_target_similarity(class_angles).detach().clone(),
        )

    def forward(self, embeddings, labels):
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
        if labels.numel() > 0:
            if labels.min().item() < 0:
                raise ValueError("labels must be nonnegative")
            if labels.max().item() >= self.class_angles.numel():
                raise ValueError("label exceeds the configured class count")

        embeddings = F.normalize(
            embeddings, p=2, dim=-1, eps=1e-8
        )
        count = embeddings.size(0)
        if count < 2:
            return embeddings.sum() * 0.0

        predicted = torch.matmul(embeddings, embeddings.t())
        target_table = self.target_similarity.to(
            device=predicted.device,
            dtype=predicted.dtype,
        )
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
        return pairwise[off_diagonal].mean()


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
):
    for name, value in {
        "fusion_ce_weight": fusion_ce_weight,
        "unimodal_ce_weight": unimodal_ce_weight,
        "distillation_weight": distillation_weight,
        "circular_weight": circular_weight,
    }.items():
        if value < 0:
            raise ValueError("{} must be nonnegative".format(name))

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

    circular = outputs["fusion_logits"].sum() * 0.0
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
        circular = circular_loss_function(
            valid_embeddings, valid_labels
        )

    unimodal_ce = text_ce + audio_ce + visual_ce
    distillation = text_kl + audio_kl + visual_kl
    total = (
        fusion_ce_weight * fusion_ce
        + unimodal_ce_weight * unimodal_ce
        + distillation_weight * distillation
        + circular_weight * circular
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
        "circular_cse": circular,
    }
