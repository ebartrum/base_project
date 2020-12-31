from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Encoder:
    img_size: int = "${expt.train.img_size}"

@dataclass
class Decoder:
    img_size: int = "${expt.train.img_size}"

@dataclass
class StandardEncoder(Encoder):
    _target_: str = "core.models.StandardEncoder"

@dataclass
class StandardDecoder(Decoder):
    _target_: str = "core.models.StandardDecoder"
