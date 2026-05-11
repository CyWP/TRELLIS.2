import torch

from torch import Tensor


class CompletionSamplerMixin:
    """
    A mixin class that forces constrained regions to fit a target.
    """

    def set_target(self, target, mask):
        self.target = target
        self.target_mask = mask

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
        return self.target * self.target_mask + out * (1 - self.target_mask)
