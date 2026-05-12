import torch
import numpy as np
from easydict import EasyDict as edict
from torch import Tensor
from typing import Union, Optional, Any
from tqdm import tqdm

from ...modules.sparse.basic import (
    SparseTensor,
    sparse_cat,
)


class InpaintSamplerMixin:
    """
    Sparse/dense inpainting mixin for flow matching samplers.

    Semantics:
        region == 1  -> FREE / inpainting region
        region == 0  -> constrained to target

    Dense:
        x = region * sample + (1-region) * target

    Sparse:
        region sparse coords define FREE voxels.
        All other target voxels are constrained.

    Sparse implementation therefore constructs:

        constrained_target
            UNION
        evolving_free_sample

    every iteration.
    """

    # ============================================================
    # API
    # ============================================================

    def set_target(
        self,
        target: Union[Tensor, SparseTensor],
        region: Union[Tensor, SparseTensor],
    ):
        self.inpaint_target = target
        self.inpaint_region = region

        self._inpaint_cache = None

    # ============================================================
    # Coordinate hashing
    # ============================================================

    def _sparse_coord_hash(
        self,
        coords: Tensor,
    ) -> Tensor:

        coords = coords.long()

        base = 2048

        strides = base ** torch.arange(
            coords.shape[1],
            device=coords.device,
            dtype=torch.long,
        )

        return (coords * strides[None]).sum(dim=1)

    # ============================================================
    # Sparse cache construction
    # ============================================================

    def _build_sparse_inpaint_cache(
        self,
        sample: SparseTensor,
    ):

        target = self.inpaint_target
        region = self.inpaint_region

        target_hash = self._sparse_coord_hash(target.coords)

        region_hash = self._sparse_coord_hash(region.coords)

        sample_hash = self._sparse_coord_hash(sample.coords)

        # ========================================================
        # FREE REGION
        # sample ∩ region
        # ========================================================

        sorted_region_hash, _ = torch.sort(region_hash)

        pos = torch.searchsorted(
            sorted_region_hash,
            sample_hash,
        )

        inside_region = (pos < len(sorted_region_hash)) & (
            sorted_region_hash[pos.clamp(max=len(sorted_region_hash) - 1)]
            == sample_hash
        )

        free_sample_idx = inside_region.nonzero().squeeze(1)

        # ========================================================
        # FIXED TARGET
        # target \ region
        # ========================================================

        pos2 = torch.searchsorted(
            sorted_region_hash,
            target_hash,
        )

        target_inside_region = (pos2 < len(sorted_region_hash)) & (
            sorted_region_hash[pos2.clamp(max=len(sorted_region_hash) - 1)]
            == target_hash
        )

        constrained_target_idx = (~target_inside_region).nonzero().squeeze(1)

        fixed_target = target.replace(
            target.feats[constrained_target_idx],
            target.coords[constrained_target_idx],
        )
        breakpoint()
        self._inpaint_cache = {
            # evolving free subset
            "free_sample_idx": free_sample_idx,
            # static constrained subset
            "fixed_target": fixed_target,
        }

    # ============================================================
    # Dense projection
    # ============================================================

    def _apply_dense_inpaint(
        self,
        x: Tensor,
    ) -> Tensor:

        return (
            self.inpaint_region * x + (1.0 - self.inpaint_region) * self.inpaint_target
        )

    # ============================================================
    # Sparse projection
    # ============================================================

    def _apply_sparse_inpaint(
        self,
        x: SparseTensor,
    ) -> SparseTensor:

        if self._inpaint_cache is None:
            self._build_sparse_inpaint_cache(x)

        cache = self._inpaint_cache

        # --------------------------------------------------------
        # Dynamic free region from evolving sample
        # --------------------------------------------------------

        free_sample = x.replace(
            x.feats[cache["free_sample_idx"]],
            x.coords[cache["free_sample_idx"]],
        )

        fixed_target = cache["fixed_target"]

        # --------------------------------------------------------
        # Direct concatenation
        # --------------------------------------------------------

        feats = torch.cat(
            [
                fixed_target.feats,
                free_sample.feats,
            ],
            dim=0,
        )

        coords = torch.cat(
            [
                fixed_target.coords,
                free_sample.coords,
            ],
            dim=0,
        )

        return SparseTensor(
            feats=feats,
            coords=coords,
        )

    # ============================================================
    # Generic projection
    # ============================================================

    def _apply_inpaint(
        self,
        x,
    ):

        if self.inpaint_target is None:
            return x

        if isinstance(x, Tensor):
            return self._apply_dense_inpaint(x)

        elif isinstance(x, SparseTensor):
            return self._apply_sparse_inpaint(x)

        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    # ============================================================
    # Model inference
    # ============================================================

    def _inference_model(
        self,
        model,
        x_t,
        t,
        cond,
        guidance_strength,
        guidance_interval,
        **kwargs,
    ):
        return super()._inference_model(
            model,
            x_t,
            t,
            cond,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            **kwargs,
        )

    # ============================================================
    # Sampling
    # ============================================================

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ):

        sample = noise

        # --------------------------------------------------------
        # Build sparse cache once
        # --------------------------------------------------------

        if (
            isinstance(sample, SparseTensor)
            and self.inpaint_target is not None
            and self._inpaint_cache is None
        ):
            self._build_sparse_inpaint_cache(sample)

        # --------------------------------------------------------
        # Initial projection
        # --------------------------------------------------------

        sample = self._apply_inpaint(sample)

        # --------------------------------------------------------
        # Time schedule
        # --------------------------------------------------------

        t_seq = np.linspace(
            1,
            0,
            steps + 1,
        )

        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        t_seq = t_seq.tolist()

        t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]

        ret = edict(
            {
                "samples": None,
                "pred_x_t": [],
            }
        )

        # --------------------------------------------------------
        # Sampling loop
        # --------------------------------------------------------

        for t, t_prev in tqdm(
            t_pairs,
            desc=tqdm_desc,
            disable=not verbose,
        ):

            # ----------------------------------------------------
            # Model prediction
            # ----------------------------------------------------

            v = self._inference_model(
                model,
                sample,
                t,
                cond,
                **kwargs,
            )

            # ----------------------------------------------------
            # Euler update
            # ----------------------------------------------------

            sample = sample - (t - t_prev) * v

            # ----------------------------------------------------
            # Re-apply hard boundary condition
            # ----------------------------------------------------

            sample = self._apply_inpaint(sample)

            ret.pred_x_t.append(sample)

        ret.samples = sample

        return ret
