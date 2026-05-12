import torch
import numpy as np
from easydict import EasyDict as edict
from torch import Tensor
from typing import Union, Optional, Any
from tqdm import tqdm

from ...modules.sparse.basic import SparseTensor


class InpaintSamplerMixin:
    """
    Soft-constrained inpainting for both dense Tensor and SparseTensor.

    Dense:
        x <- (1 - m) * x + m * target

    Sparse:
        same update, but only on overlapping coordinates.

    IMPORTANT:
    Sparse coordinate correspondences are computed ONCE and cached.
    """

    # ============================================================
    # Setup
    # ============================================================

    def set_target(
        self, target: Union[Tensor, SparseTensor], region: Union[Tensor, SparseTensor]
    ):
        self.inpaint_target = target
        self.inpaint_region = region

        self._inpaint_cache = None

    # ============================================================
    # Sparse coordinate hashing
    # ============================================================

    def _sparse_coord_hash(
        self,
        coords: Tensor,
    ) -> Tensor:
        """
        Vectorized integer coordinate hashing.

        Assumes coordinates are in [0, 1023].
        Supports:
            [b, x, y, z]
        """

        coords = coords.long()

        ndim = coords.shape[1]

        base = 2048

        strides = base ** torch.arange(
            ndim,
            device=coords.device,
            dtype=torch.long,
        )

        return (coords * strides[None]).sum(dim=1)

    # ============================================================
    # Sparse overlap cache
    # ============================================================

    def _build_sparse_inpaint_cache(
        self,
        sample: SparseTensor,
    ):
        """
        Precompute overlapping sparse voxel indices.

        Cache:
            sample_idx
            target_idx
        """

        sample_hash = self._sparse_coord_hash(sample.coords)

        target_hash = self._sparse_coord_hash(self.inpaint_target.coords)

        sorted_target_hash, perm = torch.sort(target_hash)

        pos = torch.searchsorted(
            sorted_target_hash,
            sample_hash,
        )

        valid = (pos < len(sorted_target_hash)) & (
            sorted_target_hash[pos.clamp(max=len(sorted_target_hash) - 1)]
            == sample_hash
        )

        sample_idx = valid.nonzero().squeeze(1)

        target_idx = perm[pos[valid]]

        self._inpaint_cache = {
            "sample_idx": sample_idx,
            "target_idx": target_idx,
        }

    # ============================================================
    # Dense projection
    # ============================================================

    def _apply_dense_inpaint(
        self,
        x: Tensor,
    ) -> Tensor:

        m = self.inpaint_region

        return (1.0 - m) * x + m * self.inpaint_target

    # ============================================================
    # Sparse projection
    # ============================================================

    def _apply_sparse_inpaint(
        self,
        x: SparseTensor,
    ) -> SparseTensor:

        if self._inpaint_cache is None:
            self._build_sparse_inpaint_cache(x)

        sample_idx = self._inpaint_cache["sample_idx"]
        target_idx = self._inpaint_cache["target_idx"]

        if len(sample_idx) == 0:
            return x

        new_feats = x.feats.clone()

        target_feats = self.inpaint_target.feats[target_idx]
        mask_feats = self.inpaint_region.feats[target_idx]

        new_feats[sample_idx] = (1.0 - mask_feats) * new_feats[
            sample_idx
        ] + mask_feats * target_feats

        return x.replace(new_feats)

    # ============================================================
    # Generic projection
    # ============================================================

    def _apply_inpaint(
        self,
        x: Union[Tensor, SparseTensor],
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
        """
        IMPORTANT:
        Never modify the velocity field directly.
        """

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
        """
        Euler flow sampling with soft inpainting constraints.
        """

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
        # Euler integration
        # --------------------------------------------------------

        for t, t_prev in tqdm(
            t_pairs,
            desc=tqdm_desc,
            disable=not verbose,
        ):

            # ----------------------------------------------------
            # Predict velocity
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
            # Soft state-space projection
            # ----------------------------------------------------

            sample = self._apply_inpaint(sample)

            ret.pred_x_t.append(sample)

        ret.samples = sample

        return ret
