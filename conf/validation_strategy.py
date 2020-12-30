from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any

@dataclass
class ValidationStrategy:
    _target_: str = MISSING

@dataclass
class StandardValidationStrategy(ValidationStrategy):
    _target_: str = "core.strategy.standard.validation"
