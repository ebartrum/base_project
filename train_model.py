import hydra
from conf.config import Config
import conf.experiments.experiments
from omegaconf import OmegaConf
import os

@hydra.main(config_name="config")
def main(cfg: Config) -> None:
    with open("options_used.yaml", "w") as f: 
        f.write(OmegaConf.to_yaml(cfg)) 
    #lazy import executed on singularity image
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from core.figures.callback import FiguresCallback
    from core.lightning_module import LM
    
    lightning_module = LM(cfg.expt)
    logger = TensorBoardLogger(save_dir="logs", name='',
            version='')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        filepath='logs',
        mode='min')
    full_checkpoint = get_full_ckpt(cfg)
    trainer = pl.Trainer(gpus=cfg.expt.submission.gpus,
            max_epochs=cfg.expt.train.num_epochs,
            logger=logger, checkpoint_callback=checkpoint_callback\
                    if cfg.expt.submission.save_checkpoints else None,
                    callbacks=[FiguresCallback(cfg.expt.figures)],
            resume_from_checkpoint=full_checkpoint,
            fast_dev_run=cfg.expt.submission.fast_dev_run)    
    trainer.fit(lightning_module)  

def get_full_ckpt(cfg):
    if cfg.expt.submission.resume_expt:
        ckpt_dir = [x[0] for x in os.walk(os.path.join(
            hydra.utils.get_original_cwd(), "outputs"))
            if x[0].endswith(f"/{cfg.expt.submission.resume_expt}")][0]
        ckpt = os.path.join(ckpt_dir, "logs.ckpt")
        return ckpt
    if cfg.expt.submission.checkpoint:
        ckpt = os.path.join(hydra.utils.get_original_cwd(),
                cfg.expt.submission.checkpoint)
        return ckpt
    
if __name__ == "__main__":
    main()

