from dataclasses import dataclass, field
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from typing import List, Any, Optional
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.conf import SweepDir
from hydra.core.config_store import ConfigStore
from conf.training_strategy import TrainingStrategy, StandardTrainingStrategy
from conf.validation_strategy import ValidationStrategy, StandardValidationStrategy
from conf.data import BaseData, StandardData, DatasetCollection
from conf.models import Encoder, Decoder, StandardEncoder, StandardDecoder
from conf import figure

cs = ConfigStore.instance()

@dataclass
class Submission:
    run_type: str = "slurm"
    gpus: int = 1
    save_checkpoints: bool = True
    checkpoint: Optional[str] = None
    resume_expt: Optional[str] = None
    fast_dev_run: bool = False

@dataclass
class Slurm:
    log_dir: str = "logs/slurm"
    partition: str = MISSING
    time: int = MISSING
    sing_img: str = "~/Documents/singularity_images/mega_kaolin.simg"

@dataclass
class SmallSlurm(Slurm):
    partition: str = "small"
    time: int = 4320 #72 hours * 60 mins

@dataclass
class DevelSlurm(Slurm):
    partition: str = "devel"
    time: int = 60

@dataclass
class Launcher(SlurmQueueConf):
    partition: str = "${expt.slurm.partition}"
    _target_: str = "hydra_plugins.sing_launcher.SingLauncher"
    comment: str = "${expt.slurm.sing_img}"
    timeout_min: str = "${expt.slurm.time}"
cs.store(group="hydra/launcher", name="submitit_slurm_sing", node=Launcher)

@dataclass
class Sweep(SweepDir):
    dir: str = "./outputs/${now:%Y-%m-%d/}"
    subdir: str = "${hydra.job.id}/${hydra.job.num}"
cs.store(group="hydra/sweep", name="custom_sweep", node=Sweep)

@dataclass
class Train:
    batch_size: int = 8
    num_epochs: int = 99999
    lr: float = 1e-5
    img_size: int = 128

@dataclass
class LossWeight:
    recon: float = 1

@dataclass
class BaseExperiment:
    submission: Submission = Submission()
    slurm: Slurm = SmallSlurm()
    train: Train = Train()
    data: BaseData = MISSING
    datasets: DatasetCollection = MISSING
    loss_weight: LossWeight = LossWeight()
    figures: List[figure.Base] = field(
            default_factory=lambda: [figure.KaggleTurntable()])
    training_strategy: TrainingStrategy = StandardTrainingStrategy()
    validation_strategy: ValidationStrategy = StandardValidationStrategy()
    encoder: Encoder = MISSING
    decoder: Decoder = MISSING

defaults = [
    {"expt": MISSING},
    {"hydra/launcher": "submitit_slurm_sing"},
    {"hydra/sweep": "custom_sweep"},
    {"expt.encoder": "standard"},
    {"expt.decoder": "standard"},
    {"expt.slurm": "small"}
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    expt: Any = MISSING

cs.store(name="config", node=Config)
cs.store(group="expt.slurm", name="small", node=SmallSlurm)
cs.store(group="expt.slurm", name="devel", node=DevelSlurm)
cs.store(group="expt.encoder", name="standard", node=StandardEncoder)
cs.store(group="expt.decoder", name="standard", node=StandardDecoder)
