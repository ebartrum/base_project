import pytorch_lightning as pl
import torch

def training(batch, batch_nb, optimizer_idx, pl_module):
    img = batch
    encoding = pl_module.encoder(img)
    recon = pl_module.decoder(encoding)
    recon_loss = torch.mean((recon - img)**2)
    loss = recon_loss
    pl_module.log('train_loss', loss, prog_bar=True)
    return loss

def validation(batch, batch_nb, pl_module):
    img = batch
    recon = pl_module.forward(img)
    recon_loss = torch.mean((recon - img)**2)
    loss = recon_loss
    return loss
