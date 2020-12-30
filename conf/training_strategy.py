from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Any

@dataclass
class TrainingStrategy:
    _target_: str = MISSING

@dataclass
class StandardTrainingStrategy(TrainingStrategy):
    _target_: str = "core.strategy.standard.training"
