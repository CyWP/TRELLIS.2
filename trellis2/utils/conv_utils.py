from __future__ import annotations

import torch

from typing import Optional, Union, Tuple
from torch.nn import Conv3d

from ..modules.sparse import SparseConv3d, SparseTensor

IntSeq = Union[int, Tuple[int]]
FloatSeq = Union[float, Tuple[float]]


class GaussianFilter3D:

    _layouts = ["sparse", "dense", "both"]

    def __init__(
        self,
        kernel_size: IntSeq,
        device: torch.device,
        channels: int = 1,
        sigma: Optional[FloatSeq] = None,
        layout: str = "both",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        if layout not in self._layouts:
            raise ValueError(f"Layout must be chosen from {self._layouts}.")
        self.layout = layout
        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) == 3:
            k = tuple(kernel_size)
        else:
            raise ValueError(
                "Kernel size must either be an integer or a tuple of length 3."
            )
        if any(ks % 2 == 0 for ks in k):
            raise ValueError("Kernel sizes must be odd.")
        if sigma is None:
            s = (ks / 3 for ks in k)
        elif isinstance(sigma, int):
            s = (sigma, sigma, sigma)
        elif len(sigma) == 3:
            s = sigma
        else:
            raise ValueError("Sigma must either be a float or a tuple of length 3.")
        w = self.kernel_weights(k, s)
        if self.layout != "dense":
            self._init_sparse_conv(w, k, channels)
        if self.layout != "sparse":
            self._init_dense_conv(w, k, channels)

    def kernel_weights(self, kernel_size: IntSeq, sigma: FloatSeq) -> torch.Tensor:
        k1ds = []
        for k, s in zip(kernel_size, sigma):
            r = k // 2
            co = torch.arange(-r, r + 1, device=self.device, dtype=self.dtype)
            k1d = torch.exp(-(co**2) / (2 * s**2))
            k1ds.append(k1d / k1d.sum())
        return k1ds[0][:, None, None] * k1ds[1][None, :, None] * k1ds[2][None, None, :]

    def _init_dense_conv(
        self, weight: torch.Tensor, kernel_size: IntSeq, channels: int
    ):
        pad = (k // 2 for k in kernel_size)
        kw = weight.view(1, 1, *kernel_size)
        self.dense_module = Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.dense_module.weight.data = kw.repeat(channels, channels, 1, 1, 1)
        self.dense_module.requires_grad_(False)

    def _init_sparse_conv(
        self, weight: torch.Tensor, kernel_size: IntSeq, channels: int
    ):
        kw = weight.view(1, *kernel_size, 1)
        self.sparse_module = SparseConv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=None,
            bias=False,
        )
        self.sparse_module.to(self.dtype)
        self.sparse_module.to(self.device)
        self.sparse_module.weight.data = kw.repeat(channels, 1, 1, 1, channels)
        self.sparse_module.requires_grad_(False)

    def forward(
        self, x: Union[torch.Tensor, SparseTensor]
    ) -> Union[torch.Tensor, SparseTensor]:
        if isinstance(x, SparseTensor):
            if self.layout == "dense":
                raise ValueError(
                    "Module was initialized as dense only and cannot process sparse tensors."
                )
            x.coords = x.coords.to(torch.int32).contiguous()
            return self.sparse_module(x)
        else:
            if self.layout == "sparse":
                raise ValueError(
                    "Module was initialized as sparse only and cannot process dense tensor."
                )
            return self.dense_module(x)

    def __call__(
        self, x: Union[torch.Tensor, SparseTensor]
    ) -> Union[torch.Tensor, SparseTensor]:
        return self.forward(x)

    def to(self, *args) -> GaussianFilter3D:
        for arg in args:
            if isinstance(arg, torch.dtype):
                self.dtype = arg
            elif isinstance(arg, torch.device):
                self.device = self.device
            else:
                raise ValueError(
                    f"Method accepts torch.dtype and torch.device. Got {arg.__class__}"
                )
            if hasattr(self, "sparse_module"):
                self.sparse_module = self.sparse_module.to(arg)
            if hasattr(self, "dense_module"):
                self.dense_module = self.dense_module.to(arg)
        return self
