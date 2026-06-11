import torch
from pathlib import Path

from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline

from ..modules.sparse import SparseTensor


class InspectionPipeline(Trellis2ImageTo3DPipeline):

    @classmethod
    def from_pretrained(cls, path, config_file="pipeline_inspect.json"):
        return super().from_pretrained(path, config_file)

    def setup(self, parent: Path):
        self.shape_slat_path = parent / "shape_slat"
        self.shape_slat_path.mkdir(exist_ok=True, parents=True)
        self.shape_slat_sampler.set_folder(self.shape_slat_path)
        self.tex_slat_path = parent / "tex_slat"
        self.tex_slat_path.mkdir(exist_ok=True)
        self.tex_slat_sampler.set_folder(self.tex_slat_path)
