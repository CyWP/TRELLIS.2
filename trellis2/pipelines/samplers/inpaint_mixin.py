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
            sample = self._apply_inpaint(sample)

            out.pred_x_prev = sample

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

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
    ) -> Union[Tensor, SparseTensor]:

        self._inpaint_cache = None

        # ============================================================
        # Dense
        # ============================================================

        if isinstance(target, Tensor):

            if noise is None:
                noise = torch.randn_like(target)

            self._inpaint_cache = {
                "type": "dense",
                "region": region,
                "target": target,
            }
            return noise
            return region * noise + (1.0 - region) * target

        # ============================================================
        # Sparse
        # ============================================================

        elif isinstance(target, SparseTensor):

            if template is None:
                raise ValueError("Sparse inpainting requires template coords.")

            template = template.long().contiguous()
            target_coords = target.coords.long().contiguous()
            region_coords = region.coords.long().contiguous()

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
                sorted_region_hash[
                    tmpl_pos_region.clamp(max=len(sorted_region_hash) - 1)
                ]
                == template_hash
            )

            # outside editable region
            constrained_template_idx = (~tmpl_inside_region).nonzero().squeeze(1)

            constrained_template_hash = template_hash[constrained_template_idx]

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

            constrained_feats = target.feats[matched_target_idx].contiguous()

            # ========================================================
            # Build initial sample
            # ========================================================

            feat_dim = target.feats.shape[1]

            if noise is None:

                feats = torch.randn(
                    template.shape[0],
                    feat_dim,
                    device=target.feats.device,
                    dtype=target.feats.dtype,
                )

            else:

                feats = noise.feats.clone()

            # initialize constrained region from target
            # feats[constrained_template_idx] = constrained_feats

            # feats = feats.contiguous()

            sample = SparseTensor(
                feats=feats,
                coords=template.contiguous(),
            )

            # ========================================================
            # Cache projection info
            self._inpaint_cache = {
                "type": "sparse",
                "constrained_idx": (constrained_template_idx.contiguous()),
                "constrained_feats": (constrained_feats.contiguous()),
            }

            return sample

        else:
            raise TypeError(f"Unsupported type: {type(target)}")

    # ============================================================
    # Sparse projection
    # ============================================================

    def _apply_sparse_inpaint(
        self,
        x: SparseTensor,
    ) -> SparseTensor:

        cache = self._inpaint_cache

        feats = x.feats.clone()

        feats[cache["constrained_idx"]] = cache["constrained_feats"]
        breakpoint()
        return SparseTensor(
            feats=feats.contiguous(),
            coords=x.coords.contiguous(),
        )

    # ============================================================
    # Dense projection
    # ============================================================

    def _apply_dense_inpaint(
        self,
        x: Tensor,
    ) -> Tensor:

        cache = self._inpaint_cache

        return cache["region"] * x + (1.0 - cache["region"]) * cache["target"]

    # ============================================================
    # Generic projection
    # ============================================================

    def _apply_inpaint(
        self,
        x: Union[Tensor, SparseTensor],
    ) -> Union[Tensor, SparseTensor]:

        if self._inpaint_cache is None:
            return x

        if isinstance(x, Tensor):
            return self._apply_dense_inpaint(x)

        elif isinstance(x, SparseTensor):
            return self._apply_sparse_inpaint(x)

        else:
            raise TypeError(f"Unsupported type: {type(x)}")
