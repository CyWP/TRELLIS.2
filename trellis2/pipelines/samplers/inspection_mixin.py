from __future__ import annotations

from typing import Optional, Union, Any

import numpy as np
import torch
import json
from easydict import EasyDict as edict
from torch import Tensor
from tqdm import tqdm
from pathlib import Path

from ...modules.sparse.basic import SparseTensor


class InspectionMixin:
    # Just add this mixin in the sampler class when you want to inspect output
    def set_folder(self, folder: Path):
        self.parent_folder = folder
        self.num_steps = 0
        self.should_save = True

    def toggle_save(self, val: bool):
        self.should_save = val

    def save_tensor(self, name: str, tensor: torch.Tensor | SparseTensor):
        tensor_folder = self.parent_folder / name
        tensor_folder.mkdir(exist_ok=True)
        if isinstance(tensor, SparseTensor):
            torch.save(tensor.feats.clone().cpu(), tensor_folder / "feats.pt")
            torch.save(tensor.coords.clone().cpu(), tensor_folder / "coords.pt")
        else:
            torch.save(tensor.clone().cpu(), tensor_folder / "data.pt")

    def save_dict(self, name: str, data: dict):
        save_folder = self.parent_folder / name
        save_folder.mkdir(exist_ok=True)
        with open(save_folder / "data.json", "w") as f:
            json.dump(data, f, indent=2)

    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs
    ):
        preds = super().sample_once(model, x_t, t, t_prev, cond, **kwargs)
        if self.should_save:
            step_str = f"{self.num_steps:03}"
            self.save_tensor(step_str, preds["pred_x_0"])
            self.save_dict(step_str, {"t": t})
        self.num_steps += 1
        return preds

    @torch.no_grad()
    def optimize_noise(self, *args, **kwargs):
        old_val = self.should_save
        self.toggle_save(False)
        ret = super().optimize_noise(*args, **kwargs)
        self.toggle_save(old_val)
        return ret
