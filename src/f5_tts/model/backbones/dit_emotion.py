"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNormZero_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    EmotionFiLM,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


class TextEmbedding(nn.Module):
    def __init__(
        self, emotion_condition_type, text_num_embeds, text_dim, conv_layers=0, conv_mult=2,
        emotion_num_embeds=6, emotion_dim=None,
    ):
        super().__init__()
        self.emotion_condition_type = emotion_condition_type
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)

        if self.emotion_condition_type == "text_early_fusion":
            emotion_dim = text_dim
            self.emotion_embeder = nn.Embedding(emotion_num_embeds + 1, emotion_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], emotion, seq_len, drop_text=False, drop_emotion=False):  # noqa: F722
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.emotion_condition_type == "text_early_fusion":
            emotion = emotion + 1
            emotion = emotion[:, :seq_len]
            batch, emotion_len = emotion.shape[0], emotion.shape[1]
            emotion = F.pad(emotion, (0, seq_len - emotion_len), value=0)

            if drop_emotion:
                emotion = torch.zeros_like(emotion)

            emotion = self.emotion_embeder(emotion)
            text += 0.10 * emotion

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)

        return text


class EmotionEmbedding(nn.Module):
    def __init__(self, emotion_num_embeds, emotion_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.emotion_embeder = nn.Embedding(emotion_num_embeds + 1, emotion_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(emotion_dim, self.precompute_max_pos), persistent=False
            )
            self.emotion_blocks = nn.Sequential(
                *[ConvNeXtV2Block(emotion_dim, emotion_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, emotion: int["b nt"], seq_len, drop_emotion=False):  # noqa: F722
        emotion = emotion + 1
        emotion = emotion[:, :seq_len]
        batch, emotion_len = emotion.shape[0], emotion.shape[1]
        emotion = F.pad(emotion, (0, seq_len - emotion_len), value=0)

        if drop_emotion:
            emotion = torch.zeros_like(emotion)

        emotion = self.emotion_embeder(emotion)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            emotion_pos_embed = self.freqs_cis[pos_idx]
            emotion = emotion + emotion_pos_embed
            emotion = self.emotion_blocks(emotion)

        return emotion


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, emotion_dim, out_dim, emotion_condition_type, load_emotion_weights=True):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.proj_emotion = nn.Linear(mel_dim * 2 + text_dim + emotion_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        self.weights_setup = False
        self.emotion_condition_type = emotion_condition_type
        self.load_emotion_weights = load_emotion_weights

    def _initialize_proj_emotion_weights(self, init_type, weight_reduction_scale=1):
        with torch.no_grad():
            self.proj_emotion.weight[:, : self.proj.weight.shape[1]] = self.proj.weight

            if init_type == "zeros":
                self.proj_emotion.weight[:, self.proj.weight.shape[1] :] = 0
            elif init_type == "xavier_reduced":
                self.proj_emotion.weight[:, self.proj.weight.shape[1] :] = (
                    weight_reduction_scale
                    * torch.nn.init.xavier_uniform_(
                        torch.empty_like(self.proj_emotion.weight[:, self.proj.weight.shape[1] :])
                    )
                )

            if self.proj.bias is not None:
                self.proj_emotion.bias = torch.nn.Parameter(self.proj.bias.clone())
        self.weights_setup = True

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
        emotion_embed=None,
        drop_audio_cond=False,
    ):
        if not self.weights_setup and self.emotion_condition_type == "text_mirror":
            if self.load_emotion_weights:
                raise RuntimeError(
                    "InputEmbedding weights not initialized. Call _initialize_proj_emotion_weights() first."
                )

        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        if self.emotion_condition_type in ["no_emotion_condition", "text_early_fusion", "film"]:
            x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        elif self.emotion_condition_type == "text_mirror":
            x = self.proj_emotion(torch.cat((x, cond, text_embed, emotion_embed), dim=-1))
        else:
            raise NotImplementedError(f"emotion_condition_type {self.emotion_condition_type} is not implemented")

        x = self.conv_pos_embed(x) + x
        return x


class DiTConditioned(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        emotion_num_embeds=6,
        text_dim=None,
        emotion_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        emotion_conditioning={},
    ):
        super().__init__()

        self.emotion_conditioning = emotion_conditioning

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
            emotion_dim = mel_dim
        self.text_embed = TextEmbedding(
            self.emotion_conditioning["emotion_condition_type"],
            text_num_embeds,
            text_dim,
            conv_layers=conv_layers,
            emotion_num_embeds=emotion_num_embeds,
            emotion_dim=emotion_dim,
        )
        self.emotion_embed = EmotionEmbedding(emotion_num_embeds, emotion_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(
            mel_dim,
            text_dim,
            emotion_dim,
            dim,
            emotion_condition_type=self.emotion_conditioning["emotion_condition_type"],
            load_emotion_weights=self.emotion_conditioning.get("load_emotion_weights", False),
        )

        if self.emotion_conditioning["emotion_condition_type"] == "film":
            self.emotion_film_blocks = nn.ModuleList([EmotionFiLM(dim, emotion_dim) for _ in range(depth)])

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text: int["b nt"],  # noqa: F722
        emotion,
        time: float["b"] | float[""],  # noqa: F821 F722
        drop_audio_cond,
        drop_text,
        drop_emotion_cond,
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        text_embed = self.text_embed(text, emotion, seq_len, drop_text=drop_text, drop_emotion=drop_emotion_cond)

        if self.emotion_conditioning["emotion_condition_type"] in ["no_emotion_condition", "text_early_fusion"]:
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
            emotion_embed = None
        elif self.emotion_conditioning["emotion_condition_type"] == "film":
            emotion_embed = self.emotion_embed(emotion, seq_len, drop_emotion=drop_emotion_cond)
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        elif self.emotion_conditioning["emotion_condition_type"] == "text_mirror":
            emotion_embed = self.emotion_embed(emotion, seq_len, drop_emotion=drop_emotion_cond)
            x = self.input_embed(x, cond, text_embed, emotion_embed, drop_audio_cond=drop_audio_cond)
        else:
            raise NotImplementedError(
                f'emotion_condition_type {self.emotion_conditioning["emotion_condition_type"]} is not implemented'
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, t, mask=mask, rope=rope)
            if emotion_embed is not None and self.emotion_conditioning["emotion_condition_type"] == "film":
                x = self.emotion_film_blocks[i](x, emotion_embed)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
