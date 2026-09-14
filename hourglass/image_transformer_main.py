"""k-diffusion transformer diffusion models, version 3.

This variant keeps the original main hourglass path over concatenated
segmentation/image inputs, and adds a separate conditioning-image encoder.
The conditioning features are fused into the main encoder with learnable
linear interpolation at each encoder scale before the main bottleneck.
"""

import torch
from torch import nn

from .axial_rope import make_axial_pos
from .image_transformer_base import (
    FourierFeatures,
    GlobalAttentionSpec,
    GlobalTransformerLayer,
    Level,
    LevelSpec,
    Linear,
    MappingNetwork,
    MappingSpec,
    NeighborhoodAttentionSpec,
    NeighborhoodTransformerLayer,
    RMSNorm,
    TokenMerge,
    TokenSplit,
    TokenSplitWithoutSkip,
    apply_wd,
    downscale_pos,
    filter_params,
    tag_module,
)


class LerpFeatureFuse(nn.Module):
    """Learnable interpolation fuse: lerp(main, cond_feat, alpha)."""

    def __init__(self, init=0.5):
        super().__init__()
        self.fac = nn.Parameter(torch.ones(1) * init)

    def forward(self, main, cond_feat):
        return torch.lerp(main, cond_feat, self.fac.to(main.dtype))


class ImageConditionEncoderV3(nn.Module):
    """Conditioning encoder branch with L-1 levels (no bottleneck)."""

    def __init__(self, levels, mapping_width, patch_size, cond_in_channels):
        super().__init__()
        self.mapping_width = mapping_width
        self.patch_in = TokenMerge(cond_in_channels, levels[0].width, patch_size)

        self.down_levels = nn.ModuleList()
        for i, spec in enumerate(levels[:-1]):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping_width,
                    dropout=spec.dropout,
                )
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping_width,
                    spec.self_attn.kernel_size,
                    dropout=spec.dropout,
                )
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")
            self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))

        # Only transitions between the first L-1 levels are needed.
        self.merges = nn.ModuleList(
            [TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-2], levels[1:-1])]
        )

    def forward(self, cond_img, pos):
        x = cond_img.movedim(-3, -1)
        x = self.patch_in(x)
        # Keep this branch independent of rectified-flow time conditioning.
        cond = torch.zeros(x.shape[0], self.mapping_width, device=x.device, dtype=x.dtype)

        level_features = []
        for i, down_level in enumerate(self.down_levels):
            x = down_level(x, pos, cond)
            level_features.append(x)
            if i < len(self.merges):
                x = self.merges[i](x)
                pos = downscale_pos(pos)

        return level_features


class ImageTransformerDenoiserModelV3(nn.Module):
    """V3 hourglass denoiser with separate conditioning-image encoder fusion."""

    def __init__(
        self,
        levels,
        mapping,
        in_channels,
        out_channels,
        patch_size,
        num_classes=0,
        mapping_cond_dim=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_in_channels = in_channels - out_channels

        if self.cond_in_channels <= 0:
            raise ValueError(
                "in_channels must be greater than out_channels so cond_img channels can be inferred"
            )

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(
            MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout),
            "mapping",
        )

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping.width,
                    dropout=spec.dropout,
                )
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(
                    spec.width,
                    spec.d_ff,
                    spec.self_attn.d_head,
                    mapping.width,
                    spec.self_attn.kernel_size,
                    dropout=spec.dropout,
                )
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")

            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList(
            [TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])]
        )
        self.splits = nn.ModuleList(
            [TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])]
        )

        self.cond_encoder = ImageConditionEncoderV3(
            levels=levels,
            mapping_width=mapping.width,
            patch_size=patch_size,
            cond_in_channels=self.cond_in_channels,
        )
        self.encoder_fuses = nn.ModuleList([LerpFeatureFuse() for _ in range(len(levels) - 1)])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0},
        ]
        return groups

    def forward(self, x, t, cond_img=None):
        if cond_img is None:
            raise ValueError("cond_img is required for ImageTransformerDenoiserModelV3")
        if cond_img.shape[1] != self.cond_in_channels:
            raise ValueError(
                f"cond_img has {cond_img.shape[1]} channels, expected {self.cond_in_channels}"
            )

        x = torch.cat((x, cond_img), dim=1)
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"concatenated input has {x.shape[1]} channels, expected {self.in_channels}"
            )

        # Main path over concatenated segmentation + conditioning image.
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2)

        time_emb = self.time_in_proj(self.time_emb(t[..., None]))
        cond = self.mapping(time_emb)

        cond_features = self.cond_encoder(cond_img, pos)

        # Hourglass encoder with scale-wise fusion from condition encoder.
        skips, poses = [], []
        for down_level, merge, cond_feat, fuse in zip(
            self.down_levels,
            self.merges,
            cond_features,
            self.encoder_fuses,
        ):
            x = down_level(x, pos, cond)
            x = fuse(x, cond_feat)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)

        for up_level, split, skip, pos in reversed(list(zip(self.up_levels, self.splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)
        return x


__all__ = [
    "GlobalAttentionSpec",
    "NeighborhoodAttentionSpec",
    "LevelSpec",
    "MappingSpec",
    "ImageTransformerDenoiserModelV3",
]
