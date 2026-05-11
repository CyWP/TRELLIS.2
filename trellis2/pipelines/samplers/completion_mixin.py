import torch

from torch import Tensor
from typing import Union, Optional

from ...modules.sparse.basic import SparseTensor


class CompletionSamplerMixin:
    """
    A mixin class that forces constrained regions to fit a target.
    """

    def set_target(
        self,
        target: Union[Tensor, SparseTensor],
        mask: Optional[Union[Tensor, SparseTensor]] = None,
    ):
        self.target = target
        if isinstance(target, Tensor):
            self.target_mask = (
                (self.target != 0.0).to(target.dtype) if mask is None else mask
            )
        breakpoint()

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
        out = super()._inference_model(
            model,
            x_t,
            t,
            cond,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            **kwargs,
        )
        if self.target is None:
            return out
        elif isinstance(self.target, Tensor):
            return self.target * self.target_mask + out * (1 - self.target_mask)
        else:
            oco = out.coords
            tco = self.target.coords
            eqs = (oco[:, None, :] == tco[None, :, :]).all(dim=2)
            breakpoint()
