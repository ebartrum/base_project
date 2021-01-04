from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional
from conf.data import DatasetCollection

@dataclass
class Config:
    dir: str = "figures"
    # data_root: str = "${expt.data.test_root}"
    # data_total_poses: str = "${expt.data.total_poses}"
    # mask_data_root: str = "${expt.data.test_mask_root}"
    # file_extn: str = "${expt.data.file_extn}"
    img_size: str = "${expt.train.img_size}"
    # black_bg: str = "${expt.data.black_bg}"
    datasets: DatasetCollection = "${expt.datasets}"
    filename: Optional[str] = None

@dataclass
class Base:
    _target_: MISSING
    cfg: Config = Config()

@dataclass
class RainbowSquare(Base):
    _target_: str = "core.figures.types.RainbowSquare"

@dataclass
class ReconGrid(Base):
    ncol: int = 5
    _target_: str = "core.figures.types.ReconGrid"

@dataclass
class InputReconInputGrid(Base):
    ncol: int = 5
    _target_: str = "core.figures.types.InputReconInputGrid"
