from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Encoder:
    pass

@dataclass
class Decoder:
    pass

@dataclass
class StandardEncoder(Encoder):
    _target_: str = "core.models.StandardEncoder"

@dataclass
class StandardDecoder(Decoder):
    _target_: str = "core.models.StandardDecoder"
