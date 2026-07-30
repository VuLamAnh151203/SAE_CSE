import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_MODES = (
    "sdt",
    "sdt_cosine",
    "sdt_cse",
    "sdt_cse_all_cosine",
    "sdt_cse_all_modal_cse",
    "sdt_cse_fusion_only",
    "sdt_cse_learnable_angles",
    "sdt_cse_learnable_angles_confusion_gap",
    "sdt_cse_confusion_margin",
    "sdt_cse_bilevel_confusion_gap",
)
CIRCULAR_CSE_MODES = (
    "sdt_cse",
    "sdt_cse_all_cosine",
    "sdt_cse_all_modal_cse",
    "sdt_cse_fusion_only",
    "sdt_cse_learnable_angles",
    "sdt_cse_learnable_angles_confusion_gap",
    "sdt_cse_confusion_margin",
    "sdt_cse_bilevel_confusion_gap",
)
ALL_COSINE_MODES = (
    "sdt_cse_all_cosine",
    "sdt_cse_all_modal_cse",
)
ALL_MODAL_CSE_MODES = ("sdt_cse_all_modal_cse",)
FUSION_ONLY_MODES = ("sdt_cse_fusion_only",)
CONFUSION_GAP_MODES = (
    "sdt_cse_learnable_angles_confusion_gap",
)
CONFUSION_MARGIN_MODES = (
    "sdt_cse_confusion_margin",
    "sdt_cse_bilevel_confusion_gap",
)
BILEVEL_ANGLE_MODES = ("sdt_cse_bilevel_confusion_gap",)
LEARNABLE_ANGLE_MODES = (
    "sdt_cse_learnable_angles",
    *CONFUSION_GAP_MODES,
)
CIRCLE_ORDER = (0, 4, 3, 5, 1, 2)
SDT_RESIDUAL_UPDATES = ("standard", "spherical")


class LearnableCircularAngles(nn.Module):
    """Order-preserving circular angles initialized from a fixed prior."""

    def __init__(
        self,
        num_classes=6,
        prior_angles=None,
        circle_order=CIRCLE_ORDER,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if len(circle_order) != num_classes:
            raise ValueError(
                "circle_order length must equal num_classes"
            )
        if sorted(int(index) for index in circle_order) != list(
            range(num_classes)
        ):
            raise ValueError(
                "circle_order must be a permutation of class IDs"
            )

        order = torch.tensor(circle_order, dtype=torch.long)
        inverse_order = torch.argsort(order)
        if prior_angles is None:
            ordered_prior = (
                torch.arange(num_classes, dtype=torch.float32)
                * (2.0 * math.pi / num_classes)
            )
            prior_angles = ordered_prior[inverse_order]
        else:
            prior_angles = torch.as_tensor(
                prior_angles, dtype=torch.float32
            ).detach().clone()
        if prior_angles.shape != (num_classes,):
            raise ValueError(
                "prior_angles must have shape [num_classes]"
            )
        if not torch.isfinite(prior_angles).all():
            raise ValueError(
                "prior_angles must contain only finite values"
            )

        two_pi = prior_angles.new_tensor(2.0 * math.pi)
        anchor_label = int(order[0])
        anchored_prior = torch.remainder(
            prior_angles - prior_angles[anchor_label],
            two_pi,
        )
        ordered_prior = anchored_prior[order]
        if not torch.isclose(
            ordered_prior[0],
            ordered_prior.new_zeros(()),
            atol=1e-6,
        ):
            raise ValueError("the first circle class must anchor at zero")
        prior_gaps = torch.cat(
            (
                ordered_prior[1:] - ordered_prior[:-1],
                two_pi.unsqueeze(0) - ordered_prior[-1:],
            )
        )
        if torch.any(prior_gaps <= 0):
            raise ValueError(
                "prior angles do not follow the configured circle order"
            )

        raw_gaps = prior_gaps + torch.log(
            -torch.expm1(-prior_gaps)
        )
        self.num_classes = int(num_classes)
        self.register_buffer("circle_order", order)
        self.register_buffer("inverse_circle_order", inverse_order)
        self.register_buffer("prior_angles", anchored_prior)
        self.register_buffer("prior_gaps", prior_gaps)
        self.raw_gaps = nn.Parameter(raw_gaps)

    def normalized_gaps(self):
        positive_gaps = F.softplus(self.raw_gaps)
        return (
            2.0
            * math.pi
            * positive_gaps
            / positive_gaps.sum()
        )

    def forward(self):
        gaps = self.normalized_gaps()
        ordered_angles = torch.cat(
            (
                gaps.new_zeros(1),
                torch.cumsum(gaps[:-1], dim=0),
            )
        )
        return ordered_angles[self.inverse_circle_order]

    def angle_offsets(self, angles=None):
        if angles is None:
            angles = self()
        return angles - self.prior_angles

    def regularization(self, angles=None):
        return self.angle_offsets(angles).pow(2).sum()

    def geometry(self):
        angles = self()
        return {
            "angles": angles,
            "gaps": self.normalized_gaps(),
            "offsets": self.angle_offsets(angles),
            "regularization": self.regularization(angles),
        }


class BilevelConfusionGapAngles(nn.Module):
    """A validation-learned shared gap with fixed circular order."""

    def __init__(
        self,
        minimum_gap_degrees=70.0,
        maximum_gap_degrees=110.0,
        initial_gap_degrees=90.0,
        circle_order=CIRCLE_ORDER,
    ):
        super().__init__()
        minimum = float(minimum_gap_degrees)
        maximum = float(maximum_gap_degrees)
        initial = float(initial_gap_degrees)
        for name, value in (
            ("minimum_gap_degrees", minimum),
            ("maximum_gap_degrees", maximum),
            ("initial_gap_degrees", initial),
        ):
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
        if not 0.0 < minimum < initial < maximum < 180.0:
            raise ValueError(
                "bilevel gaps must satisfy 0 < minimum < initial "
                "< maximum < 180"
            )
        if 2.0 * maximum >= 360.0:
            raise ValueError(
                "maximum gap must leave positive remaining gaps"
            )
        if tuple(int(index) for index in circle_order) != CIRCLE_ORDER:
            raise ValueError(
                "bilevel confusion gaps require IEMOCAP circle order "
                "{}".format(CIRCLE_ORDER)
            )

        fraction = (initial - minimum) / (maximum - minimum)
        raw_initial = math.log(fraction / (1.0 - fraction))
        order = torch.tensor(circle_order, dtype=torch.long)
        self.register_buffer("circle_order", order)
        self.register_buffer(
            "inverse_circle_order", torch.argsort(order)
        )
        self.register_buffer(
            "minimum_gap_radians",
            torch.tensor(math.radians(minimum)),
        )
        self.register_buffer(
            "maximum_gap_radians",
            torch.tensor(math.radians(maximum)),
        )
        self.register_buffer(
            "initial_gap_radians",
            torch.tensor(math.radians(initial)),
        )
        self.raw_gaps = nn.Parameter(torch.tensor(raw_initial))

        with torch.no_grad():
            initial_angles = self.forward()
        self.register_buffer("prior_angles", initial_angles.clone())

    def confusion_gap(self):
        fraction = torch.sigmoid(self.raw_gaps)
        return self.minimum_gap_radians + fraction * (
            self.maximum_gap_radians - self.minimum_gap_radians
        )

    def normalized_gaps(self):
        confusion_gap = self.confusion_gap()
        remaining_gap = (
            confusion_gap.new_tensor(2.0 * math.pi)
            - 2.0 * confusion_gap
        ) / 4.0
        return torch.stack(
            (
                confusion_gap,
                remaining_gap,
                confusion_gap,
                remaining_gap,
                remaining_gap,
                remaining_gap,
            )
        )

    def forward(self):
        gaps = self.normalized_gaps()
        ordered_angles = torch.cat(
            (
                gaps.new_zeros(1),
                torch.cumsum(gaps[:-1], dim=0),
            )
        )
        return ordered_angles[self.inverse_circle_order]

    def angle_offsets(self, angles=None):
        if angles is None:
            angles = self()
        return angles - self.prior_angles

    def regularization(self, angles=None):
        return self.angle_offsets(angles).pow(2).sum()

    def geometry(self):
        angles = self()
        return {
            "angles": angles,
            "gaps": self.normalized_gaps(),
            "offsets": self.angle_offsets(angles),
            "regularization": self.regularization(angles),
        }


def gelu(x):
    """GELU used by the original SDT implementation."""
    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (x + 0.044715 * torch.pow(x, 3))
        )
    )


class SphericalResidualUpdate(nn.Module):
    """Learned normalized residual interpolation for valid utterances."""

    def __init__(
        self,
        hidden_dim,
        initial_alpha=0.1,
        eps=1e-8,
    ):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < float(initial_alpha) < 1.0:
            raise ValueError(
                "initial_alpha must be strictly between 0 and 1"
            )
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        initial_logit = math.log(
            float(initial_alpha) / (1.0 - float(initial_alpha))
        )
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.alpha_logit = nn.Parameter(
            torch.tensor([initial_logit], dtype=torch.float32)
        )
        self._diagnostics_enabled = False
        self.reset_diagnostics()

    @property
    def effective_alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def reset_diagnostics(self):
        self._diagnostic_count = 0
        self._diagnostic_norm_sum = 0.0
        self._diagnostic_norm_square_sum = 0.0
        self._diagnostic_norm_min = float("inf")
        self._diagnostic_norm_max = -float("inf")
        self._diagnostic_angle_sum = 0.0

    def enable_diagnostics(self, enabled=True, reset=True):
        if reset:
            self.reset_diagnostics()
        self._diagnostics_enabled = bool(enabled)

    def diagnostic_summary(self):
        count = self._diagnostic_count
        if count == 0:
            return {
                "valid_state_count": 0,
                "effective_alpha": float(
                    self.effective_alpha.detach().cpu().item()
                ),
                "output_norm_mean": None,
                "output_norm_std": None,
                "output_norm_min": None,
                "output_norm_max": None,
                "mean_angular_movement_degrees": None,
            }
        mean = self._diagnostic_norm_sum / count
        variance = max(
            self._diagnostic_norm_square_sum / count - mean * mean,
            0.0,
        )
        return {
            "valid_state_count": int(count),
            "effective_alpha": float(
                self.effective_alpha.detach().cpu().item()
            ),
            "output_norm_mean": float(mean),
            "output_norm_std": float(math.sqrt(variance)),
            "output_norm_min": float(self._diagnostic_norm_min),
            "output_norm_max": float(self._diagnostic_norm_max),
            "mean_angular_movement_degrees": float(
                self._diagnostic_angle_sum / count
            ),
        }

    def forward(self, current, proposal, padding_mask=None):
        if current.shape != proposal.shape:
            raise ValueError(
                "current and proposal must have identical shapes"
            )
        if current.ndim != 3:
            raise ValueError(
                "current and proposal must have shape [B, L, H]"
            )
        if current.size(-1) != self.hidden_dim:
            raise ValueError(
                "last dimension must equal configured hidden_dim"
            )
        if not torch.isfinite(current).all():
            raise ValueError("current must contain only finite values")
        if not torch.isfinite(proposal).all():
            raise ValueError("proposal must contain only finite values")
        if padding_mask is None:
            padding_mask = torch.zeros(
                current.shape[:-1],
                dtype=torch.bool,
                device=current.device,
            )
        else:
            if padding_mask.shape != current.shape[:-1]:
                raise ValueError(
                    "padding_mask must have shape [B, L]"
                )
            padding_mask = padding_mask.to(
                device=current.device,
                dtype=torch.bool,
            )

        current_direction = F.normalize(
            current,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        proposal_direction = F.normalize(
            proposal,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        alpha = self.effective_alpha.to(
            device=current.device,
            dtype=current.dtype,
        )
        mixed = (
            (1.0 - alpha) * current_direction
            + alpha * proposal_direction
        )
        mixed_norm = torch.linalg.vector_norm(
            mixed, ord=2, dim=-1, keepdim=True
        )
        safe_mixed = torch.where(
            mixed_norm > self.eps,
            mixed,
            current_direction,
        )
        output = F.normalize(
            safe_mixed,
            p=2,
            dim=-1,
            eps=self.eps,
        )
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self._diagnostics_enabled:
            with torch.no_grad():
                valid = ~padding_mask
                valid_output = output[valid]
                if valid_output.numel() > 0:
                    norms = torch.linalg.vector_norm(
                        valid_output, ord=2, dim=-1
                    )
                    current_valid = current_direction[valid]
                    cosine = torch.sum(
                        current_valid * valid_output, dim=-1
                    ).clamp(-1.0, 1.0)
                    angles = torch.rad2deg(torch.acos(cosine))
                    count = int(norms.numel())
                    self._diagnostic_count += count
                    self._diagnostic_norm_sum += float(
                        norms.sum().cpu().item()
                    )
                    self._diagnostic_norm_square_sum += float(
                        norms.square().sum().cpu().item()
                    )
                    self._diagnostic_norm_min = min(
                        self._diagnostic_norm_min,
                        float(norms.min().cpu().item()),
                    )
                    self._diagnostic_norm_max = max(
                        self._diagnostic_norm_max,
                        float(norms.max().cpu().item()),
                    )
                    self._diagnostic_angle_sum += float(
                        angles.sum().cpu().item()
                    )
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def propose(self, x):
        intermediate = self.dropout_1(gelu(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(intermediate))

    def forward(self, x):
        return self.propose(x) + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError("model_dim must be divisible by head_count")
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        shape = (
            batch_size,
            -1,
            self.head_count,
            self.dim_per_head,
        )
        key = self.linear_k(key).view(*shape).transpose(1, 2)
        value = self.linear_v(value).view(*shape).transpose(1, 2)
        query = self.linear_q(query).view(*shape).transpose(1, 2)
        query = query / math.sqrt(self.dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)
        attention = self.softmax(scores)
        context = torch.matmul(self.dropout(attention), value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.model_dim)
        )
        return self.linear(context)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, speaker_embedding):
        return x + self.pe[:, : x.size(1)] + speaker_embedding


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        residual_update="standard",
        attention_alpha_init=0.1,
        mlp_alpha_init=0.1,
    ):
        super().__init__()
        if residual_update not in SDT_RESIDUAL_UPDATES:
            raise ValueError(
                "residual_update must be one of {}".format(
                    SDT_RESIDUAL_UPDATES
                )
            )
        self.residual_update = residual_update
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.attention_update = None
        self.mlp_update = None
        if residual_update == "spherical":
            self.attention_update = SphericalResidualUpdate(
                d_model,
                initial_alpha=attention_alpha_init,
            )
            self.mlp_update = SphericalResidualUpdate(
                d_model,
                initial_alpha=mlp_alpha_init,
            )

    def forward(self, iteration, inputs_a, inputs_b, mask, self_attention):
        residual_current = inputs_b
        attention_query = (
            self.layer_norm(inputs_b)
            if iteration != 0
            else inputs_b
        )
        attention_mask = mask.unsqueeze(1)
        if self_attention:
            context = self.self_attn(
                attention_query,
                attention_query,
                attention_query,
                mask=attention_mask,
            )
        else:
            context = self.self_attn(
                inputs_a,
                inputs_a,
                attention_query,
                mask=attention_mask,
            )
        attention_proposal = self.dropout(context)
        if self.residual_update == "standard":
            attention_result = attention_proposal + attention_query
            return self.feed_forward(attention_result)

        attention_result = self.attention_update(
            residual_current,
            attention_proposal,
            padding_mask=mask,
        )
        mlp_proposal = self.feed_forward.propose(attention_result)
        return self.mlp_update(
            attention_result,
            mlp_proposal,
            padding_mask=mask,
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        heads,
        layers,
        dropout=0.1,
        residual_update="standard",
        attention_alpha_init=0.1,
        mlp_alpha_init=0.1,
    ):
        super().__init__()
        if residual_update not in SDT_RESIDUAL_UPDATES:
            raise ValueError(
                "residual_update must be one of {}".format(
                    SDT_RESIDUAL_UPDATES
                )
            )
        self.residual_update = residual_update
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    residual_update=residual_update,
                    attention_alpha_init=attention_alpha_init,
                    mlp_alpha_init=mlp_alpha_init,
                )
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize_valid(states, padding_mask):
        states = F.normalize(states, p=2, dim=-1, eps=1e-8)
        return states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        x_a,
        x_b,
        mask,
        speaker_embedding,
        self_attention=False,
    ):
        padding_mask = mask.eq(0)
        if self_attention:
            x_b = self.dropout(self.pos_emb(x_b, speaker_embedding))
            if self.residual_update == "spherical":
                x_b = self._normalize_valid(x_b, padding_mask)
            for index, layer in enumerate(self.transformer_inter):
                x_b = layer(
                    index,
                    x_b,
                    x_b,
                    padding_mask,
                    self_attention=True,
                )
            return x_b

        x_a = self.dropout(self.pos_emb(x_a, speaker_embedding))
        x_b = self.dropout(self.pos_emb(x_b, speaker_embedding))
        if self.residual_update == "spherical":
            x_a = self._normalize_valid(x_a, padding_mask)
            x_b = self._normalize_valid(x_b, padding_mask)
        for index, layer in enumerate(self.transformer_inter):
            x_b = layer(
                index,
                x_a,
                x_b,
                padding_mask,
                self_attention=False,
            )
        return x_b


class UnimodalGatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset="IEMOCAP"):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == "MELD":
            self.fc.weight.data.copy_(torch.eye(hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, feature):
        return torch.sigmoid(self.fc(feature)) * feature


class MultimodalGatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, *features):
        if len(features) < 2:
            raise ValueError(
                "MultimodalGatedFusion requires at least two features"
            )
        stacked_features = torch.stack(features, dim=-2)
        gate_logits = torch.stack(
            [self.fc(feature) for feature in features],
            dim=-2,
        )
        weights = self.softmax(gate_logits)
        return torch.sum(weights * stacked_features, dim=-2)


class SphericalFusionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(input_dim, embedding_dim)

    def forward(self, fusion_features):
        projected = self.linear_1(fusion_features)
        projected = gelu(projected)
        projected = self.dropout(projected)
        projected = self.linear_2(projected)
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)


class CosineEmotionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes=6,
        initial_scale=16.0,
    ):
        super().__init__()
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        self.class_weights = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
        self.log_scale = nn.Parameter(
            torch.tensor(float(initial_scale)).log()
        )

    @property
    def effective_scale(self):
        return self.log_scale.exp().clamp(min=1.0, max=100.0)

    def cosine_scores(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        class_weights = F.normalize(
            self.class_weights, p=2, dim=-1, eps=1e-8
        )
        return torch.matmul(embeddings, class_weights.t())

    def forward(self, embeddings):
        return self.effective_scale * self.cosine_scores(embeddings)


class SDTCSEModel(nn.Module):
    """SDT encoder with original or spherical/cosine fusion heads."""

    def __init__(
        self,
        d_text=1024,
        d_visual=342,
        d_audio=1582,
        n_head=8,
        n_classes=6,
        hidden_dim=1024,
        n_speakers=2,
        dropout=0.5,
        experiment_mode="sdt_cse",
        embedding_dim=256,
        projection_dropout=0.1,
        initial_cosine_scale=16.0,
        initial_class_angles=None,
        bilevel_gap_minimum_degrees=70.0,
        bilevel_gap_maximum_degrees=110.0,
        bilevel_gap_initial_degrees=90.0,
        sdt_residual_update="standard",
        spherical_attention_alpha_init=0.1,
        spherical_mlp_alpha_init=0.1,
    ):
        super().__init__()
        if experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(
                "experiment_mode must be one of {}".format(EXPERIMENT_MODES)
            )
        if n_speakers not in (2, 9):
            raise ValueError("n_speakers must be 2 or 9")
        if sdt_residual_update not in SDT_RESIDUAL_UPDATES:
            raise ValueError(
                "sdt_residual_update must be one of {}".format(
                    SDT_RESIDUAL_UPDATES
                )
            )
        for name, value in (
            (
                "spherical_attention_alpha_init",
                spherical_attention_alpha_init,
            ),
            (
                "spherical_mlp_alpha_init",
                spherical_mlp_alpha_init,
            ),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(
                    "{} must be strictly between 0 and 1".format(
                        name
                    )
                )

        self.experiment_mode = experiment_mode
        self.sdt_residual_update = sdt_residual_update
        self.spherical_attention_alpha_init = float(
            spherical_attention_alpha_init
        )
        self.spherical_mlp_alpha_init = float(
            spherical_mlp_alpha_init
        )
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.circular_angle_learner = None
        if experiment_mode in BILEVEL_ANGLE_MODES:
            self.circular_angle_learner = BilevelConfusionGapAngles(
                minimum_gap_degrees=bilevel_gap_minimum_degrees,
                maximum_gap_degrees=bilevel_gap_maximum_degrees,
                initial_gap_degrees=bilevel_gap_initial_degrees,
                circle_order=CIRCLE_ORDER,
            )
        elif experiment_mode in LEARNABLE_ANGLE_MODES:
            self.circular_angle_learner = LearnableCircularAngles(
                num_classes=n_classes,
                prior_angles=initial_class_angles,
                circle_order=CIRCLE_ORDER,
            )
        padding_idx = n_speakers

        self.speaker_embeddings = nn.Embedding(
            n_speakers + 1,
            hidden_dim,
            padding_idx=padding_idx,
        )

        self.textf_input = nn.Conv1d(
            d_text, hidden_dim, kernel_size=1, bias=False
        )
        self.acouf_input = nn.Conv1d(
            d_audio, hidden_dim, kernel_size=1, bias=False
        )
        self.visuf_input = nn.Conv1d(
            d_visual, hidden_dim, kernel_size=1, bias=False
        )

        self.t_t = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.a_t = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.v_t = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.a_a = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.t_a = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.v_a = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.v_v = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.t_v = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )
        self.a_v = TransformerEncoder(
            hidden_dim,
            hidden_dim,
            n_head,
            1,
            dropout,
            residual_update=sdt_residual_update,
            attention_alpha_init=spherical_attention_alpha_init,
            mlp_alpha_init=spherical_mlp_alpha_init,
        )

        self.t_t_gate = UnimodalGatedFusion(hidden_dim)
        self.a_t_gate = UnimodalGatedFusion(hidden_dim)
        self.v_t_gate = UnimodalGatedFusion(hidden_dim)
        self.a_a_gate = UnimodalGatedFusion(hidden_dim)
        self.t_a_gate = UnimodalGatedFusion(hidden_dim)
        self.v_a_gate = UnimodalGatedFusion(hidden_dim)
        self.v_v_gate = UnimodalGatedFusion(hidden_dim)
        self.t_v_gate = UnimodalGatedFusion(hidden_dim)
        self.a_v_gate = UnimodalGatedFusion(hidden_dim)

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)
        self.last_gate = MultimodalGatedFusion(hidden_dim)

        self.text_projector = None
        self.audio_projector = None
        self.visual_projector = None
        if experiment_mode in ALL_COSINE_MODES:
            self.text_projector = SphericalFusionHead(
                hidden_dim,
                embedding_dim=embedding_dim,
                dropout=projection_dropout,
            )
            self.audio_projector = SphericalFusionHead(
                hidden_dim,
                embedding_dim=embedding_dim,
                dropout=projection_dropout,
            )
            self.visual_projector = SphericalFusionHead(
                hidden_dim,
                embedding_dim=embedding_dim,
                dropout=projection_dropout,
            )
            self.t_output_layer = CosineEmotionClassifier(
                embedding_dim,
                num_classes=n_classes,
                initial_scale=initial_cosine_scale,
            )
            self.a_output_layer = CosineEmotionClassifier(
                embedding_dim,
                num_classes=n_classes,
                initial_scale=initial_cosine_scale,
            )
            self.v_output_layer = CosineEmotionClassifier(
                embedding_dim,
                num_classes=n_classes,
                initial_scale=initial_cosine_scale,
            )
        elif experiment_mode in FUSION_ONLY_MODES:
            self.t_output_layer = None
            self.a_output_layer = None
            self.v_output_layer = None
        else:
            self.t_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            self.a_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
            self.v_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        self.all_output_layer = None
        self.fusion_projector = None
        self.cosine_classifier = None
        if experiment_mode == "sdt":
            self.all_output_layer = nn.Linear(hidden_dim, n_classes)
        else:
            self.fusion_projector = SphericalFusionHead(
                hidden_dim,
                embedding_dim=embedding_dim,
                dropout=projection_dropout,
            )
            self.cosine_classifier = CosineEmotionClassifier(
                embedding_dim,
                num_classes=n_classes,
                initial_scale=initial_cosine_scale,
            )

    def _speaker_embeddings(self, qmask, dialogue_lengths):
        speaker_indices = torch.argmax(qmask, dim=-1).clone()
        for index, length in enumerate(dialogue_lengths):
            length = int(length)
            if length < speaker_indices.size(1):
                speaker_indices[index, length:] = self.n_speakers
        return self.speaker_embeddings(speaker_indices)

    def encode_fusion(
        self,
        text_features,
        visual_features,
        audio_features,
        utterance_mask,
        speaker_mask,
        dialogue_lengths,
    ):
        speaker_embeddings = self._speaker_embeddings(
            speaker_mask, dialogue_lengths
        )

        text = self.textf_input(
            text_features.permute(1, 2, 0)
        ).transpose(1, 2)
        audio = self.acouf_input(
            audio_features.permute(1, 2, 0)
        ).transpose(1, 2)
        visual = self.visuf_input(
            visual_features.permute(1, 2, 0)
        ).transpose(1, 2)

        t_t = self.t_t(
            text,
            text,
            utterance_mask,
            speaker_embeddings,
            self_attention=True,
        )
        a_t = self.a_t(
            audio, text, utterance_mask, speaker_embeddings
        )
        v_t = self.v_t(
            visual, text, utterance_mask, speaker_embeddings
        )

        a_a = self.a_a(
            audio,
            audio,
            utterance_mask,
            speaker_embeddings,
            self_attention=True,
        )
        t_a = self.t_a(
            text, audio, utterance_mask, speaker_embeddings
        )
        v_a = self.v_a(
            visual, audio, utterance_mask, speaker_embeddings
        )

        v_v = self.v_v(
            visual,
            visual,
            utterance_mask,
            speaker_embeddings,
            self_attention=True,
        )
        t_v = self.t_v(
            text, visual, utterance_mask, speaker_embeddings
        )
        a_v = self.a_v(
            audio, visual, utterance_mask, speaker_embeddings
        )

        text_hidden = self.features_reduce_t(
            torch.cat(
                [
                    self.t_t_gate(t_t),
                    self.a_t_gate(a_t),
                    self.v_t_gate(v_t),
                ],
                dim=-1,
            )
        )
        audio_hidden = self.features_reduce_a(
            torch.cat(
                [
                    self.a_a_gate(a_a),
                    self.t_a_gate(t_a),
                    self.v_a_gate(v_a),
                ],
                dim=-1,
            )
        )
        visual_hidden = self.features_reduce_v(
            torch.cat(
                [
                    self.v_v_gate(v_v),
                    self.t_v_gate(t_v),
                    self.a_v_gate(a_v),
                ],
                dim=-1,
            )
        )
        fusion_features = self.last_gate(
            text_hidden, audio_hidden, visual_hidden
        )
        return {
            "text_representation": text_hidden,
            "audio_representation": audio_hidden,
            "visual_representation": visual_hidden,
            "fusion_features": fusion_features,
        }

    def forward(
        self,
        text_features,
        visual_features,
        audio_features,
        utterance_mask,
        speaker_mask,
        dialogue_lengths,
    ):
        representations = self.encode_fusion(
            text_features,
            visual_features,
            audio_features,
            utterance_mask,
            speaker_mask,
            dialogue_lengths,
        )
        text_hidden = representations["text_representation"]
        audio_hidden = representations["audio_representation"]
        visual_hidden = representations["visual_representation"]
        fusion_features = representations["fusion_features"]

        embeddings = None
        fusion_cosine_scores = None
        if self.experiment_mode == "sdt":
            fusion_logits = self.all_output_layer(fusion_features)
        else:
            embeddings = self.fusion_projector(fusion_features)
            fusion_cosine_scores = (
                self.cosine_classifier.cosine_scores(embeddings)
            )
            fusion_logits = (
                self.cosine_classifier.effective_scale
                * fusion_cosine_scores
            )

        text_embeddings = None
        audio_embeddings = None
        visual_embeddings = None
        if self.experiment_mode in ALL_COSINE_MODES:
            text_embeddings = self.text_projector(text_hidden)
            audio_embeddings = self.audio_projector(audio_hidden)
            visual_embeddings = self.visual_projector(visual_hidden)
            text_logits = self.t_output_layer(text_embeddings)
            audio_logits = self.a_output_layer(audio_embeddings)
            visual_logits = self.v_output_layer(visual_embeddings)
        elif self.experiment_mode in FUSION_ONLY_MODES:
            text_logits = None
            audio_logits = None
            visual_logits = None
        else:
            text_logits = self.t_output_layer(text_hidden)
            audio_logits = self.a_output_layer(audio_hidden)
            visual_logits = self.v_output_layer(visual_hidden)

        angle_state = (
            self.circular_angle_learner.geometry()
            if self.circular_angle_learner is not None
            else None
        )
        return {
            "text_representation": text_hidden,
            "audio_representation": audio_hidden,
            "visual_representation": visual_hidden,
            "fusion_features": fusion_features,
            "embeddings": embeddings,
            "text_embeddings": text_embeddings,
            "audio_embeddings": audio_embeddings,
            "visual_embeddings": visual_embeddings,
            "unimodal_circular_cse_enabled": (
                self.experiment_mode in ALL_MODAL_CSE_MODES
            ),
            "confusion_gap_enabled": (
                self.experiment_mode in CONFUSION_GAP_MODES
            ),
            "confusion_margin_enabled": (
                self.experiment_mode in CONFUSION_MARGIN_MODES
            ),
            "fusion_logits": fusion_logits,
            "fusion_cosine_scores": fusion_cosine_scores,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "visual_logits": visual_logits,
            "class_angles": (
                None if angle_state is None else angle_state["angles"]
            ),
            "angle_gaps": (
                None if angle_state is None else angle_state["gaps"]
            ),
            "angle_offsets": (
                None if angle_state is None else angle_state["offsets"]
            ),
            "angle_regularization": (
                None
                if angle_state is None
                else angle_state["regularization"]
            ),
        }

    def current_circular_angle_state(self):
        if self.circular_angle_learner is None:
            return None
        return self.circular_angle_learner.geometry()

    def named_spherical_residual_updates(self):
        return [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, SphericalResidualUpdate)
        ]

    def spherical_residual_parameters(self):
        return [
            module.alpha_logit
            for _, module in self.named_spherical_residual_updates()
        ]

    def spherical_residual_gate_state(self):
        return {
            name: module.effective_alpha
            for name, module in self.named_spherical_residual_updates()
        }

    def enable_spherical_residual_diagnostics(
        self,
        enabled=True,
        reset=True,
    ):
        for _, module in self.named_spherical_residual_updates():
            module.enable_diagnostics(enabled=enabled, reset=reset)

    def spherical_residual_diagnostics(self):
        return {
            name: module.diagnostic_summary()
            for name, module in self.named_spherical_residual_updates()
        }

    @property
    def effective_cosine_scale(self):
        if self.cosine_classifier is None:
            return None
        return self.cosine_classifier.effective_scale

    @property
    def effective_unimodal_cosine_scales(self):
        if self.experiment_mode not in ALL_COSINE_MODES:
            return {
                "text": None,
                "audio": None,
                "visual": None,
            }
        return {
            "text": self.t_output_layer.effective_scale,
            "audio": self.a_output_layer.effective_scale,
            "visual": self.v_output_layer.effective_scale,
        }
