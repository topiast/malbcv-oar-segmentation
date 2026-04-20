"""MaskMed-style architecture with FSAD fusion and set prediction head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_norm(num_channels: int, norm_name: str) -> nn.Module:
    normalized = norm_name.lower()
    if normalized == "batch":
        return nn.BatchNorm3d(num_channels)
    if normalized == "group":
        num_groups = 8 if num_channels % 8 == 0 else 4
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return nn.InstanceNorm3d(num_channels, affine=True)


class ConvNormAct(nn.Module):
    """3D convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            _make_norm(out_channels, norm_name),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual 3D convolutional block."""

    def __init__(self, channels: int, norm_name: str = "instance") -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, norm_name=norm_name)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm(channels, norm_name),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.conv2(self.conv1(x)) + x)


class EncoderStage(nn.Module):
    """Convolutional encoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        downsample: bool,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        if downsample:
            blocks.append(ConvNormAct(in_channels, out_channels, stride=2, norm_name=norm_name))
        else:
            blocks.append(ConvNormAct(in_channels, out_channels, norm_name=norm_name))
        blocks.append(ResidualBlock(out_channels, norm_name=norm_name))
        blocks.append(ResidualBlock(out_channels, norm_name=norm_name))
        self.stage = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


class DecoderStage(nn.Module):
    """UNet-like decoder merge block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_name: str) -> None:
        super().__init__()
        self.merge = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels, norm_name=norm_name),
            ResidualBlock(out_channels, norm_name=norm_name),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.merge(torch.cat([x, skip], dim=1))


class FeedForward3D(nn.Module):
    """Point-wise FFN used inside the FSAD transformer block."""

    def __init__(self, channels: int, expansion: int = 4, norm_name: str = "instance") -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            _make_norm(channels, norm_name),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FSADAttention3D(nn.Module):
    """
    Full-Scale Aware Deformable Attention.

    Queries come from one of the lower-resolution encoder stages. Values come from all
    encoder stages. Query-conditioned offsets and attention weights sample sparse points
    from the full feature hierarchy using `grid_sample`, following the paper's design.
    """

    def __init__(
        self,
        query_channels: int,
        value_channels: list[int],
        hidden_dim: int,
        num_heads: int,
        num_points: int,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_scales = len(value_channels)
        self.head_dim = hidden_dim // num_heads
        self.offset_scale = 0.35

        self.query_proj = nn.Conv3d(query_channels, hidden_dim, kernel_size=1, bias=False)
        self.value_projs = nn.ModuleList(
            [nn.Conv3d(ch, hidden_dim, kernel_size=1, bias=False) for ch in value_channels]
        )
        self.offset_proj = nn.Conv3d(
            hidden_dim,
            num_heads * self.num_scales * num_points * 3,
            kernel_size=1,
        )
        self.weight_proj = nn.Conv3d(
            hidden_dim,
            num_heads * self.num_scales * num_points,
            kernel_size=1,
        )
        self.output_proj = nn.Sequential(
            nn.Conv3d(hidden_dim, query_channels, kernel_size=1, bias=False),
            _make_norm(query_channels, norm_name),
        )

    def _base_grid(self, reference: Tensor) -> Tensor:
        _, _, depth, height, width = reference.shape
        z = torch.linspace(-1.0, 1.0, depth, device=reference.device, dtype=reference.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1)

    def forward(self, query_feature: Tensor, value_features: list[Tensor]) -> Tensor:
        batch_size, _, depth, height, width = query_feature.shape
        query = self.query_proj(query_feature)

        offsets = self.offset_proj(query).view(
            batch_size,
            self.num_heads,
            self.num_scales,
            self.num_points,
            3,
            depth,
            height,
            width,
        )
        offsets = torch.tanh(offsets) * self.offset_scale

        attn_weights = self.weight_proj(query).view(
            batch_size,
            self.num_heads,
            self.num_scales * self.num_points,
            depth,
            height,
            width,
        )
        attn_weights = torch.softmax(attn_weights, dim=2).view(
            batch_size,
            self.num_heads,
            self.num_scales,
            self.num_points,
            depth,
            height,
            width,
        )

        base_grid = self._base_grid(query_feature).view(1, 1, depth, height, width, 3)
        sampled_sum = query.new_zeros(batch_size, self.num_heads, self.head_dim, depth, height, width)

        for scale_index, (feature, projection) in enumerate(zip(value_features, self.value_projs, strict=True)):
            value = projection(feature).view(
                batch_size, self.num_heads, self.head_dim, *feature.shape[2:]
            )
            value = value.flatten(0, 1)

            for point_index in range(self.num_points):
                point_grid = base_grid + offsets[:, :, scale_index, point_index].permute(0, 1, 3, 4, 5, 2)
                point_grid = point_grid.reshape(batch_size * self.num_heads, depth, height, width, 3)
                sampled = F.grid_sample(
                    value,
                    point_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                sampled = sampled.view(batch_size, self.num_heads, self.head_dim, depth, height, width)
                weight = attn_weights[:, :, scale_index, point_index].unsqueeze(2)
                sampled_sum = sampled_sum + sampled * weight

        output = sampled_sum.reshape(batch_size, self.hidden_dim, depth, height, width)
        return self.output_proj(output)


class FSADTransformerBlock(nn.Module):
    """Transformer-style residual block with FSAD attention."""

    def __init__(
        self,
        query_channels: int,
        value_channels: list[int],
        hidden_dim: int,
        num_heads: int,
        num_points: int,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        self.attn = FSADAttention3D(
            query_channels=query_channels,
            value_channels=value_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_points=num_points,
            norm_name=norm_name,
        )
        self.ffn = FeedForward3D(query_channels, norm_name=norm_name)

    def forward(self, query_feature: Tensor, value_features: list[Tensor]) -> Tensor:
        x = query_feature + self.attn(query_feature, value_features)
        x = x + self.ffn(x)
        return x


class MaskedCrossAttention(nn.Module):
    """Cross-attention with optional mask bias from previous-stage masks."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, attention_bias: Tensor | None = None) -> Tensor:
        batch_size, num_queries, hidden_dim = queries.shape
        num_tokens = memory.shape[1]

        q = self.q_proj(queries).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_bias is not None:
            attn = attn + attention_bias.unsqueeze(1)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).reshape(batch_size, num_queries, hidden_dim)
        return self.out_proj(output)


class SegHeadTransformerBlock(nn.Module):
    """Masked cross-attention + self-attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = MaskedCrossAttention(hidden_dim, num_heads, dropout)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, attention_bias: Tensor | None) -> Tensor:
        cross = self.cross_attn(queries, memory, attention_bias)
        queries = self.norm1(queries + self.dropout(cross))

        self_attn, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm2(queries + self.dropout(self_attn))

        ff = self.linear2(self.dropout(F.gelu(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout(ff))
        return queries


class MaskedMultiScaleSegHead(nn.Module):
    """Shared-query multi-stage segmentation head from low to high resolution."""

    def __init__(
        self,
        stage_channels: list[int],
        foreground_classes: int,
        num_queries: int,
        hidden_dim: int,
        mask_dim: int,
        num_heads: int,
        dropout: float,
        attention_pool_sizes: list[tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.foreground_classes = foreground_classes
        self.no_object_index = foreground_classes
        self.attention_pool_sizes = attention_pool_sizes

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [SegHeadTransformerBlock(hidden_dim, num_heads, dropout) for _ in stage_channels]
        )
        self.token_projections = nn.ModuleList(
            [nn.Conv3d(ch, hidden_dim, kernel_size=1) for ch in stage_channels]
        )
        self.mask_feature_projections = nn.ModuleList(
            [nn.Conv3d(ch, mask_dim, kernel_size=1) for ch in stage_channels]
        )
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mask_dim),
        )
        self.class_embed = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, foreground_classes + 1),
        )

    def _pool_tokens(self, feature: Tensor, stage_index: int) -> tuple[Tensor, tuple[int, int, int]]:
        pool_size = self.attention_pool_sizes[stage_index]
        if pool_size == tuple(feature.shape[2:]):
            pooled = feature
        else:
            pooled = F.adaptive_avg_pool3d(feature, output_size=pool_size)
        tokens = pooled.flatten(2).transpose(1, 2)
        return tokens, pooled.shape[2:]

    def _attention_bias(
        self,
        prev_mask_logits: Tensor | None,
        token_shape: tuple[int, int, int],
    ) -> Tensor | None:
        if prev_mask_logits is None:
            return None
        mask_probs = torch.sigmoid(prev_mask_logits)
        pooled = F.interpolate(mask_probs, size=token_shape, mode="trilinear", align_corners=False)
        pooled = pooled.flatten(2)
        return torch.log(pooled.clamp_min(1e-4))

    def forward(self, stage_features: list[Tensor]) -> dict:
        batch_size = stage_features[0].shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        stage_outputs = []
        prev_mask_logits = None
        for stage_index, (feature, token_proj, mask_proj, block) in enumerate(
            zip(
                stage_features,
                self.token_projections,
                self.mask_feature_projections,
                self.transformer_blocks,
                strict=True,
            )
        ):
            memory_feature = token_proj(feature)
            memory_tokens, token_shape = self._pool_tokens(memory_feature, stage_index)
            attention_bias = self._attention_bias(prev_mask_logits, token_shape)

            queries = block(queries, memory_tokens, attention_bias)

            mask_embeddings = self.mask_embed(queries)
            class_logits = self.class_embed(queries)
            mask_features = mask_proj(feature)
            mask_logits = torch.einsum("bqm,bmxyz->bqxyz", mask_embeddings, mask_features)

            stage_outputs.append({
                "pred_logits": class_logits,
                "pred_masks": mask_logits,
            })
            prev_mask_logits = mask_logits

        stage_weights = [0.25, 0.5, 1.0][: len(stage_outputs)]
        if len(stage_outputs) > 3:
            stage_weights = [0.5 ** (len(stage_outputs) - index - 1) for index in range(len(stage_outputs))]
        weight_sum = sum(stage_weights)
        stage_weights = [weight / weight_sum for weight in stage_weights]

        final_logits = self._dense_semantic_logits(stage_outputs[-1]["pred_logits"], stage_outputs[-1]["pred_masks"])
        return {
            "logits": final_logits,
            "stages": stage_outputs,
            "stage_weights": stage_weights,
        }

    def _dense_semantic_logits(self, class_logits: Tensor, mask_logits: Tensor) -> Tensor:
        class_probs = torch.softmax(class_logits, dim=-1)
        foreground_probs = class_probs[..., : self.foreground_classes]
        mask_probs = torch.sigmoid(mask_logits)
        foreground_scores = torch.einsum("bqc,bqxyz->bcxyz", foreground_probs, mask_probs)

        background_score = (1.0 - foreground_scores.sum(dim=1, keepdim=True)).clamp_min(1e-6)
        probabilities = torch.cat([background_score, foreground_scores.clamp_min(1e-6)], dim=1)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probabilities.log()


@dataclass
class MaskMedConfig:
    in_channels: int
    out_channels: int
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    hidden_dim: int = 192
    mask_dim: int = 128
    num_queries: int = 13
    num_heads: int = 8
    num_points: int = 4
    dropout: float = 0.1
    norm: str = "instance"
    attention_pool_sizes: tuple[tuple[int, int, int], ...] = ((6, 6, 6), (10, 10, 10), (14, 14, 14))


class MaskMedNet(nn.Module):
    """Approximation of the paper architecture with explicit FSAD and set prediction."""

    def __init__(self, cfg: MaskMedConfig) -> None:
        super().__init__()
        if cfg.out_channels < 2:
            raise ValueError("MaskMed requires at least background + one foreground class")

        c1, c2, c3, c4 = cfg.channels
        self.foreground_classes = cfg.out_channels - 1

        self.stem = EncoderStage(cfg.in_channels, c1, downsample=False, norm_name=cfg.norm)
        self.enc2 = EncoderStage(c1, c2, downsample=True, norm_name=cfg.norm)
        self.enc3 = EncoderStage(c2, c3, downsample=True, norm_name=cfg.norm)
        self.enc4 = EncoderStage(c3, c4, downsample=True, norm_name=cfg.norm)

        self.fsad2 = FSADTransformerBlock(c2, [c1, c2, c3, c4], cfg.hidden_dim, cfg.num_heads, cfg.num_points, cfg.norm)
        self.fsad3 = FSADTransformerBlock(c3, [c1, c2, c3, c4], cfg.hidden_dim, cfg.num_heads, cfg.num_points, cfg.norm)
        self.fsad4 = FSADTransformerBlock(c4, [c1, c2, c3, c4], cfg.hidden_dim, cfg.num_heads, cfg.num_points, cfg.norm)

        self.dec3 = DecoderStage(c4, c3, c3, cfg.norm)
        self.dec2 = DecoderStage(c3, c2, c2, cfg.norm)
        self.dec1 = DecoderStage(c2, c1, c1, cfg.norm)

        self.seg_head = MaskedMultiScaleSegHead(
            stage_channels=[c3, c2, c1],
            foreground_classes=self.foreground_classes,
            num_queries=cfg.num_queries,
            hidden_dim=cfg.hidden_dim,
            mask_dim=cfg.mask_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attention_pool_sizes=list(cfg.attention_pool_sizes),
        )

        self._transformer_modules = nn.ModuleList([
            self.fsad2,
            self.fsad3,
            self.fsad4,
            self.seg_head,
        ])

    def get_param_groups(self, base_lr: float, transformer_lr_ratio: float) -> list[dict]:
        """Parameter groups for the paper's CNN vs Transformer LR ratio."""
        transformer_param_ids = {id(parameter) for module in self._transformer_modules for parameter in module.parameters()}
        transformer_params = []
        cnn_params = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in transformer_param_ids:
                transformer_params.append(parameter)
            else:
                cnn_params.append(parameter)

        return [
            {"params": cnn_params, "lr": base_lr},
            {"params": transformer_params, "lr": base_lr * transformer_lr_ratio},
        ]

    def forward(self, x: Tensor) -> dict:
        enc1 = self.stem(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        full_scale_features = [enc1, enc2, enc3, enc4]
        enc2 = self.fsad2(enc2, full_scale_features)
        enc3 = self.fsad3(enc3, full_scale_features)
        enc4 = self.fsad4(enc4, full_scale_features)

        dec3 = self.dec3(enc4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return self.seg_head([dec3, dec2, dec1])


def build_model(config: dict) -> nn.Module:
    """Build the configurable MaskMed model."""
    model_cfg = config["model"]
    cfg = MaskMedConfig(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=tuple(model_cfg.get("channels", [32, 64, 128, 256])),
        hidden_dim=model_cfg.get("hidden_dim", 192),
        mask_dim=model_cfg.get("mask_dim", 128),
        num_queries=model_cfg.get("num_queries", model_cfg["out_channels"] - 1),
        num_heads=model_cfg.get("num_heads", 8),
        num_points=model_cfg.get("num_points", 4),
        dropout=model_cfg.get("dropout", 0.1),
        norm=model_cfg.get("norm", "instance"),
        attention_pool_sizes=tuple(
            tuple(size) for size in model_cfg.get(
                "attention_pool_sizes", [[6, 6, 6], [10, 10, 10], [14, 14, 14]]
            )
        ),
    )
    return MaskMedNet(cfg)
