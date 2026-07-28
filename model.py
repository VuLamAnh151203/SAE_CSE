import math

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERIMENT_MODES = ("sdt", "sdt_cosine", "sdt_cse")


def gelu(x):
    """GELU used by the original SDT implementation."""
    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (x + 0.044715 * torch.pow(x, 3))
        )
    )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        intermediate = self.dropout_1(gelu(self.w_1(self.layer_norm(x))))
        return self.dropout_2(self.w_2(intermediate)) + x


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
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iteration, inputs_a, inputs_b, mask, self_attention):
        if iteration != 0:
            inputs_b = self.layer_norm(inputs_b)
        attention_mask = mask.unsqueeze(1)
        if self_attention:
            context = self.self_attn(
                inputs_b, inputs_b, inputs_b, mask=attention_mask
            )
        else:
            context = self.self_attn(
                inputs_a, inputs_a, inputs_b, mask=attention_mask
            )
        return self.feed_forward(self.dropout(context) + inputs_b)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super().__init__()
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_a,
        x_b,
        mask,
        speaker_embedding,
        self_attention=False,
    ):
        if self_attention:
            x_b = self.dropout(self.pos_emb(x_b, speaker_embedding))
            for index, layer in enumerate(self.transformer_inter):
                x_b = layer(
                    index,
                    x_b,
                    x_b,
                    mask.eq(0),
                    self_attention=True,
                )
            return x_b

        x_a = self.dropout(self.pos_emb(x_a, speaker_embedding))
        x_b = self.dropout(self.pos_emb(x_b, speaker_embedding))
        for index, layer in enumerate(self.transformer_inter):
            x_b = layer(
                index,
                x_a,
                x_b,
                mask.eq(0),
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

    def forward(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        class_weights = F.normalize(
            self.class_weights, p=2, dim=-1, eps=1e-8
        )
        return self.effective_scale * torch.matmul(
            embeddings, class_weights.t()
        )


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
    ):
        super().__init__()
        if experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(
                "experiment_mode must be one of {}".format(EXPERIMENT_MODES)
            )
        if n_speakers not in (2, 9):
            raise ValueError("n_speakers must be 2 or 9")

        self.experiment_mode = experiment_mode
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
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
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.a_t = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.v_t = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.a_a = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.t_a = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.v_a = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.v_v = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.t_v = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
        )
        self.a_v = TransformerEncoder(
            hidden_dim, hidden_dim, n_head, 1, dropout
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
        if self.experiment_mode == "sdt":
            fusion_logits = self.all_output_layer(fusion_features)
        else:
            embeddings = self.fusion_projector(fusion_features)
            fusion_logits = self.cosine_classifier(embeddings)

        return {
            "text_representation": text_hidden,
            "audio_representation": audio_hidden,
            "visual_representation": visual_hidden,
            "fusion_features": fusion_features,
            "embeddings": embeddings,
            "fusion_logits": fusion_logits,
            "text_logits": self.t_output_layer(text_hidden),
            "audio_logits": self.a_output_layer(audio_hidden),
            "visual_logits": self.v_output_layer(visual_hidden),
        }

    @property
    def effective_cosine_scale(self):
        if self.cosine_classifier is None:
            return None
        return self.cosine_classifier.effective_scale
