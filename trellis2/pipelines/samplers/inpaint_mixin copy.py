from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from easydict import EasyDict as edict
from torch import Tensor
from tqdm import tqdm

from ...modules.sparse.basic import SparseTensor


class InpaintSamplerMixin:
    """
    Inpainting mixin for dense and sparse samplers.

    Semantics
    ---------
    region == 1:
        FREE / editable region

    region == 0:
        constrained region
    """

    # ============================================================
    # Sampling loop
    # ============================================================

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ):
        sample = noise

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()

        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        ret = edict(
            {
                "samples": None,
                "pred_x_t": [],
                "pred_x_0": [],
            }
        )
        step = 0
        for t, t_prev in tqdm(
            t_pairs,
            desc=tqdm_desc,
            disable=not verbose,
        ):

            out = self.sample_once(
                model,
                sample,
                t,
                t_prev,
                cond,
                **kwargs,
            )

            sample = out.pred_x_prev

            # enforce constraints
            if hasattr(self, "start_step") and step >= self.start_step:
                sample = self._apply_inpaint(sample, t_prev)

            out.pred_x_prev = sample

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            step += 1

        ret.samples = sample

        return ret

    # ============================================================
    # Coordinate hashing
    # ============================================================

    def _sparse_coord_hash(
        self,
        coords: Tensor,
    ) -> Tensor:

        coords = coords.long()

        # robust enough for 64^3 / 128^3 sparse coords
        base = 4096

        strides = base ** torch.arange(
            coords.shape[1],
            device=coords.device,
            dtype=torch.long,
        )

        return (coords * strides[None]).sum(dim=1)

    # ============================================================
    # Prepare inpainting
    # ============================================================

    def prepare_inpainting(
        self,
        target: Union[Tensor, SparseTensor],
        region: Union[Tensor, SparseTensor],
        noise: Optional[Union[Tensor, SparseTensor]] = None,
        template: Optional[Tensor] = None,
        start_step: int = 0,
        hard: Optional[bool] = False,
    ) -> Union[Tensor, SparseTensor]:
        self.start_step = start_step
        self._inpaint_cache = None
        if isinstance(target, Tensor):
            return self.prepare_dense_inpainting(target, region, noise)
        elif isinstance(target, SparseTensor):
            if hard:
                return self.prepare_hard_sparse_inpainting(
                    target, region, template, noise
                )
            else:
                return self.prepare_soft_sparse_inpainting(
                    target, region, template, noise
                )
        else:
            raise TypeError(f"Unsupported type: {type(target)}")

    def prepare_dense_inpainting(
        self, target: Tensor, region: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(target)

        self._inpaint_cache = {
            "type": "dense",
            "region": region,
            "target": target,
            "noise": noise.clone(),
        }
        return noise

    def prepare_hard_sparse_inpainting(
        self,
        target: SparseTensor,
        region: Union[SparseTensor, Tensor],
        template: Tensor,
        noise: Optional[Union[SparseTensor, Tensor]] = None,
    ) -> SparseTensor:
        if template is None:
            raise ValueError("Sparse inpainting requires template coords.")

        target_coords = target.coords  # coords of target, should be subset of template
        region_coords = (
            region.coords if isinstance(region, SparseTensor) else region
        )  # Region where we want to inpaint>0, not fully subset of template

        # --------------------------------------------------------
        # Build hashes
        # --------------------------------------------------------

        template_hash = self._sparse_coord_hash(template)
        target_hash = self._sparse_coord_hash(target_coords)
        region_hash = self._sparse_coord_hash(region_coords)

        # ========================================================
        # Determine constrained template coords
        #
        # constrained =
        #     template ∩ target ∩ (~region)
        # ========================================================

        sorted_region_hash, _ = torch.sort(region_hash)

        tmpl_pos_region = torch.searchsorted(
            sorted_region_hash,
            template_hash,
        )

        tmpl_inside_region = (tmpl_pos_region < len(sorted_region_hash)) & (
            sorted_region_hash[tmpl_pos_region.clamp(max=len(sorted_region_hash) - 1)]
            == template_hash
        )  # 1d of bools, shape is len(template.co). True = idx is inside region, False is outside.

        # outside editable region
        constrained_template_idx = (
            (~tmpl_inside_region).nonzero().squeeze(1)
        )  # 1D indices of all template indices outside of region shape is len(tempate.co)-tmpl_inside_region.sum()

        # inside editable region
        free_template_idx = (tmpl_inside_region).nonzero().squeeze(1)

        constrained_template_hash = template_hash[constrained_template_idx]
        free_template_hash = template_hash[free_template_idx]

        # ========================================================
        # Match constrained template coords -> target coords
        # ========================================================

        sorted_target_hash, sorted_target_idx = torch.sort(target_hash)

        pos_target = torch.searchsorted(
            sorted_target_hash,
            constrained_template_hash,
        )

        valid_target = (pos_target < len(sorted_target_hash)) & (
            sorted_target_hash[pos_target.clamp(max=len(sorted_target_hash) - 1)]
            == constrained_template_hash
        )

        # keep only coords that actually exist in target
        constrained_template_idx = constrained_template_idx[valid_target]

        matched_target_idx = sorted_target_idx[pos_target[valid_target]]

        constrained_feats = target.feats[matched_target_idx]

        # Check matched_target_idx and what it represents
        # Find indices of region that are used,so we can keep the tensor of relevant weights

        # ========================================================
        # Build initial sample
        # ========================================================

        feat_dim = target.feats.shape[1]

        feats = (
            torch.randn(
                template.shape[0],
                feat_dim,
                device=target.feats.device,
                dtype=target.feats.dtype,
            )
            if noise is None
            else noise.feats if isinstance(noise, SparseTensor) else noise
        )

        sample = SparseTensor(
            feats=feats,
            coords=template,
        )

        # ========================================================
        # Cache projection info
        self._inpaint_cache = {
            "type": "sparse_hard",
            "constrained_idx": (constrained_template_idx),
            "constrained_feats": (constrained_feats),
            "noise": sample.feats.clone(),
        }

        return sample

    def prepare_soft_sparse_inpainting(
        self,
        target: SparseTensor,
        region: SparseTensor,
        template: SparseTensor,
        noise: Optional[SparseTensor] = None,
    ) -> SparseTensor:
        if template is None:
            raise ValueError("Sparse inpainting requires template coords.")

        template = template.long()
        target_coords = target.coords.long()
        region_coords = region.coords.long()

        feat_dim = target.feats.shape[1]

        # ========================================================
        # Build hashes for coord matching
        # ========================================================

        template_hash = self._sparse_coord_hash(template)
        target_hash = self._sparse_coord_hash(target_coords)
        region_hash = self._sparse_coord_hash(region_coords)

        # ========================================================
        # Initialize per-template-coord weights and targets
        # weight=0: fully constrained to target
        # weight=1: fully free (use sample)
        # ========================================================

        weights = torch.zeros(
            template.shape[0],
            1,
            device=target.feats.device,
            dtype=target.feats.dtype,
        )
        target_feats = torch.zeros(
            template.shape[0],
            feat_dim,
            device=target.feats.device,
            dtype=target.feats.dtype,
        )
        has_target = torch.zeros(
            template.shape[0],
            1,
            dtype=torch.bool,
            device=target.feats.device,
        )

        # ========================================================
        # Match template -> target (get target features)
        # ========================================================

        sorted_target_hash, sorted_target_idx = torch.sort(target_hash)
        pos_target = torch.searchsorted(sorted_target_hash, template_hash)
        valid_target = (pos_target < len(sorted_target_hash)) & (
            sorted_target_hash[pos_target.clamp(max=len(sorted_target_hash) - 1)]
            == template_hash
        )
        matched_template_to_target = valid_target.nonzero().squeeze(1)
        matched_target_idx = sorted_target_idx[pos_target[valid_target]]
        has_target[matched_template_to_target] = True
        target_feats[matched_template_to_target] = target.feats[matched_target_idx]

        # ========================================================
        # Match template -> region (get weights)
        # ========================================================

        sorted_region_hash, sorted_region_idx = torch.sort(region_hash)
        pos_region = torch.searchsorted(sorted_region_hash, template_hash)
        in_region = (pos_region < len(sorted_region_hash)) & (
            sorted_region_hash[pos_region.clamp(max=len(sorted_region_hash) - 1)]
            == template_hash
        )
        matched_template_to_region = in_region.nonzero().squeeze(1)
        matched_region_idx = sorted_region_idx[pos_region[in_region]]
        region_w = region.feats[matched_region_idx]
        if region_w.dim() == 1:
            region_w = region_w.unsqueeze(-1)
        weights[matched_template_to_region] = region_w

        # ========================================================
        # Free where no target available
        # ========================================================
        weights[~has_target] = 1.0

        # ========================================================
        # Build initial sample (random noise)
        # ========================================================

        feats = (
            torch.randn(
                template.shape[0],
                feat_dim,
                device=target.feats.device,
                dtype=target.feats.dtype,
            )
            if noise is None
            else noise.feats
        )

        sample = SparseTensor(
            feats=feats,
            coords=template,
        )

        # ========================================================
        # Cache for apply step
        # ========================================================
        # breakpoint()
        self._inpaint_cache = {
            "type": "sparse_soft",
            "weights": weights,
            "target_feats": target_feats,
            "noise": sample.feats.clone(),
        }

        return sample

    def _apply_sparse_inpaint(self, x: SparseTensor, t: float) -> SparseTensor:

        cache = self._inpaint_cache

        if cache["type"] == "sparse_soft":
            constrained_xt = t * cache["noise"] + (1 - t) * cache["target_feats"]
            feats = (
                cache["weights"] * x.feats + (1.0 - cache["weights"]) * constrained_xt
            )
            return SparseTensor(
                feats=feats,
                coords=x.coords,
            )
        else:
            feats = x.feats.clone()
            constrained_xt = (
                t * cache["noise"][cache["constrained_idx"]]
                + (1 - t) * cache["constrained_feats"]
            )
            feats[cache["constrained_idx"]] = constrained_xt
            return SparseTensor(
                feats=feats,
                coords=x.coords,
            )

    # ============================================================
    # Dense projection
    # ============================================================

    def _apply_dense_inpaint(self, x: Tensor, t: float) -> Tensor:

        cache = self._inpaint_cache

        constrained_xt = t * cache["noise"] + (1 - t) * cache["target"]

        return cache["region"] * x + (1 - cache["region"]) * constrained_xt

        return cache["region"] * x + (1.0 - cache["region"]) * cache["target"]

    # ============================================================
    # Generic projection
    # ============================================================

    def _apply_inpaint(
        self, x: Union[Tensor, SparseTensor], t: float
    ) -> Union[Tensor, SparseTensor]:

        if self.has_callbacks:
            for c in self.callbacks:
                x = c(x)

        if not hasattr(self, "_inpaint_cache") or self._inpaint_cache is None:
            return x

        if isinstance(x, Tensor):
            return self._apply_dense_inpaint(x, t)

        elif isinstance(x, SparseTensor):
            return self._apply_sparse_inpaint(x, t)

        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    @property
    def has_callbacks(self) -> bool:
        return hasattr(self, "callbacks") and len(self.callbacks) > 0

    def add_callback(self, func: callable):
        if not self.has_callbacks:
            self.callbacks = [func]
        else:
            self.callbacks.append(func)

    # ============================================================
    # Noise optimization via flow inversion
    # ============================================================

    @torch.no_grad()
    def optimize_noise(
        self,
        model,
        noise,
        cond=None,
        steps: int = 50,
        rescale_t: float = 1.0,
        n_loops: int = 1,
        verbose: bool = True,
        tqdm_desc: str = "Optimize",
        **kwargs,
    ):
        """
        Optimize initial noise via forward-then-invert loops.

        For each loop:
        1. Forward sample from noise → x_0_pred (with per-step inpainting)
        2. Blend x_0_pred with target at constrained positions
        3. Invert blended x_0 back to noise (no inpainting during invert)

        Returns optimized noise.
        """

        if self._inpaint_cache is None:
            return noise

        for loop_i in range(n_loops):
            # --- Forward pass: noise → x_0 ---
            sample = noise

            t_seq = np.linspace(1, 0, steps + 1)
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
            t_seq = t_seq.tolist()
            t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

            for t, t_prev in tqdm(
                t_pairs, desc=f"{tqdm_desc} Fwd", disable=not verbose
            ):
                out = self.sample_once(
                    model,
                    sample,
                    t,
                    t_prev,
                    cond,
                    **kwargs,
                )
                sample = out.pred_x_prev
                sample = self._apply_inpaint(sample, t_prev)

            x_0_pred = sample

            # --- Blend: replace constrained positions with target ---
            x_0_blended = self._apply_inpaint(x_0_pred, 0)

            # --- Inversion pass: blended x_0 → noise (no inpainting) ---
            t_pairs_inv = [(t_seq[i + 1], t_seq[i]) for i in range(steps)[::2]]

            for t, t_prev in tqdm(
                t_pairs_inv, desc=f"{tqdm_desc} Inv", disable=not verbose
            ):
                out = self.sample_once(
                    model,
                    x_0_blended,
                    t,
                    t_prev,
                    cond,
                    **kwargs,
                )
                x_0_blended = out.pred_x_prev

            noise = x_0_blended
            if isinstance(noise, SparseTensor):
                f = noise.feats
                noise.feats = (f - f.mean()) / f.std()
            else:
                noise = (noise - noise.mean()) / noise.std()
            if hasattr(self, "_inpaint_cache"):
                self._inpaint_cache["noise"] = (
                    noise.clone() if isinstance(noise, Tensor) else noise.feats.clone()
                )
        return noise
