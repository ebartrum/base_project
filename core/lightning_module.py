import pytorch_lightning as pl
from torchvision.models import vgg16_bn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torchvision
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import numpy as np

class LM(pl.LightningModule):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        self.train_dataset = hydra.utils.instantiate(cfg.datasets.train)
        self.val_dataset = hydra.utils.instantiate(cfg.datasets.val)
        self.encoder = hydra.utils.instantiate(cfg.encoder)
        self.decoder = hydra.utils.instantiate(cfg.decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
               lr=self.cfg.train.lr)
        return optimizer
   
    def training_step(self, batch, batch_nb, optimizer_idx=0):
        loss = hydra.utils.call(self.cfg.training_strategy,
                batch, batch_nb, optimizer_idx, pl_module=self)
        return loss

    def forward(self, img):
        encoding = self.encoder(img)
        recon = self.decoder(encoding)
        return recon

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                batch_size=self.cfg.train.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                batch_size=self.cfg.train.batch_size)

    def validation_step(self, batch, batch_nb):
        val_loss = hydra.utils.call(self.cfg.validation_strategy,
                batch, batch_nb, pl_module=self)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        # grid = self.create_turntable_vis(self.visualisation_data)
        # self.logger.experiment.add_image(f'Prediction', grid,
        #         self.current_epoch)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss}
