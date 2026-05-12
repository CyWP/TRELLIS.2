import torch
import numpy as np
from easydict import EasyDict as edict
from torch import Tensor
from typing import Union, Optional, Any
from tqdm import tqdm


class InpaintSamplerMixin:
    """
    Soft-constrained inpainting for flow / ODE samplers.
    Mask is float in [0, 1].
    """

    def set_target(
        self,
        target: Tensor,
        region: Tensor,  # float mask in [0, 1]
    ):
        self.inpaint_target = target
        self.inpaint_region = region

    def _apply_inpaint(self, x: Tensor) -> Tensor:
        """
        Soft projection in state space:
            x <- (1-m)x + m*target
        """
        if self.inpaint_target is None:
            return x

        m = self.inpaint_region
        return (1.0 - m) * x + m * self.inpaint_target

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
        # IMPORTANT: do not modify velocity field
        return super()._inference_model(
            model,
            x_t,
            t,
            cond,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            **kwargs,
        )

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

        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None})

        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):

            # 1. model predicts velocity
            v = self._inference_model(model, sample, t, cond, **kwargs)

            # 2. Euler step
            sample = sample - (t - t_prev) * v

            # 3. soft constraint projection
            sample = self._apply_inpaint(sample)

        ret.samples = sample
        return ret
