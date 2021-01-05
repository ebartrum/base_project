from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Encoder:
    img_size: int = "${expt.train.img_size}"
    dim_out: int = "${expt.latent_dim}"

@dataclass
class Decoder:
    img_size: int = "${expt.train.img_size}"
    dim_in: int = "${expt.latent_dim}"

@dataclass
class StandardEncoder(Encoder):
    _target_: str = "core.models.StandardEncoder"

@dataclass
class StandardDecoder(Decoder):
    _target_: str = "core.models.StandardDecoder"

@dataclass
class VGGEncoder(Encoder):
    _target_: str = "core.models.VGGEncoder"

@dataclass
class SirenDecoder(Decoder):
    _target_: str = "core.models.SirenDecoder"
