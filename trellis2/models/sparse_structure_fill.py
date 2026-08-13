from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.utils import convert_module_to, manual_cast, str_to_dtype
from ..modules.transformer import (
    AbsolutePositionEmbedder,
    ModulatedTransformerCrossBlock,
)
from ..modules.attention import RotaryPositionEmbedder
from .sparse_structure_flow import SparseStructureFlowModel, TimestepEmbedder


class SparseStructureCorrectionModel(SparseStructureFlowModel):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = "vanilla",
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(
                *[torch.arange(res, device=self.device) for res in [resolution] * 3],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
        elif pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedder(
                self.model_channels // self.num_heads, 3
            )
            coords = torch.meshgrid(
                *[torch.arange(res, device=self.device) for res in [resolution] * 3],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)

        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_layer = nn.Linear(in_channels, cond_channels)

        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], (
            f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        )
        assert x.shape == cond.shape, (
            "input, and conditioning should all ahve same shape."
        )
        assert mask.shape == x.shape[-3:], (
            "Mask shape must match spatial dims of input."
        )

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        c = cond.view(*cond.shape[:2], -1).permute(0, 2, 1).contiguous()
        m = mask.view(-1).contiguous()

        h = self.input_layer(h)[:, m, :]
        c = self.cond_layer(c)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[m][None]
            c = c + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        h = manual_cast(h, self.dtype)
        c = manual_cast(c, self.dtype)
        for block in self.blocks:
            h = block(
                h,
                t_emb,
                c,
                self.rope_phases if self.rope_phases is None else self.rope_phases[m],
            )
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        # Scatter correction back into the full velocity field.
        out = x.flatten(2).clone()  # [B, out_channels, N]

        out[:, :, m] += h.transpose(1, 2)

        out = out.view_as(x)

        return out
