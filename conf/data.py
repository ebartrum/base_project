from dataclasses import dataclass
from omegaconf import MISSING
from typing import List, Any, Optional, Callable
from torchvision import transforms

@dataclass
class BaseData:
    pass

@dataclass
class Dataset:
    pass

@dataclass
class ImageDataset(Dataset):
    _target_: str = "core.dataset.ImageDataset"
    root: str = MISSING
    img_size: int = MISSING
    file_extn: str = MISSING

@dataclass
class StandardData(BaseData):
    train_root: str = MISSING
    test_root: str = MISSING
    val_root: str = MISSING

@dataclass
class ImageData(StandardData):
    img_size: int = "${expt.train.img_size}"
    file_extn: str = "jpg"

@dataclass
class DatasetCollection:
    train: Dataset = MISSING
    test: Dataset = MISSING
    val: Dataset = MISSING

@dataclass
class ImageDatasetCollection(DatasetCollection):
    train: Dataset = ImageDataset(
            root="${expt.data.train_root}", 
            img_size="${expt.data.img_size}",
            file_extn="${expt.data.file_extn}")
    test: Dataset = ImageDataset(
            root="${expt.data.test_root}",
            img_size="${expt.data.img_size}",
            file_extn="${expt.data.file_extn}")
    val: Dataset = ImageDataset(
            root="${expt.data.val_root}",
            img_size="${expt.data.img_size}",
            file_extn="${expt.data.file_extn}")
