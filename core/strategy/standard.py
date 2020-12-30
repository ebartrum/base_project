import pytorch_lightning as pl
import torch

def training(batch, batch_nb, optimizer_idx, pl_module):
    img = batch
    encoding = pl_module.encoder(img)
    recon = pl_module.decoder(encoding)
    import ipdb;ipdb.set_trace()

    loss = compute_loss(view1, pose1_hat, geometry1['silhouette'],
            recon1, view2, geometry2['silhouette'],
            recon2, vertices, pl_module)
    pl_module.log('train_loss', loss, prog_bar=True)
    return loss

def validation(batch, batch_nb, pl_module):
    import ipdb;ipdb.set_trace()
    val_loss = pl_module.recon_loss(recon2, view2['img']) 
    return val_loss
