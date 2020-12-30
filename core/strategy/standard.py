import pytorch_lightning as pl
import torch
from core.utils import quat_rotate, quat_distance
from core.loss import iou_loss, LaplacianLoss, FlattenLoss

def training(batch, batch_nb, optimizer_idx, pl_module):
    import ipdb;ipdb.set_trace()
    view1, view2 = batch
    if optimizer_idx == 0:
        return generator_step(view1, view2, pl_module)

    if optimizer_idx == 1:
        return discriminator_step(view1, view2, pl_module)

def generator_step(view1, view2, pl_module):
    vertices, faces, pose1_hat, dist1_hat =\
            pl_module.predict_pose_shape(view1['img'])
    _, _, pose2_hat, dist2_hat =\
            pl_module.predict_pose_shape(view2['img'])
    
    geometry1 = pl_module.calc_geometric(vertices, faces, pose1_hat, dist1_hat)
    geometry2 = pl_module.calc_geometric(vertices, faces, pose2_hat, dist2_hat)

    neural_tex = pl_module.compute_nt(view1['img'], pose1_hat, dist1_hat,
            vertices, faces, geometry1)
    background = pl_module.bg_model(view1['img'])

    recon1 = pl_module.project_and_decode(neural_tex, vertices, faces,
            pose1_hat, dist1_hat, geometry1, bg=background)
    recon2 = pl_module.project_and_decode(neural_tex, vertices, faces,
            pose2_hat, dist2_hat, geometry2, bg=background)

    loss = compute_loss(view1, pose1_hat, geometry1['silhouette'],
            recon1, view2, geometry2['silhouette'],
            recon2, vertices, pl_module)
    pose_loss = (quat_distance(view1['pose'], pose1_hat) +\
            quat_distance(view2['pose'], pose2_hat)).mean()
    loss = loss + pose_loss*pl_module.cfg.loss_weight.pose

    pl_module.log('train_loss', loss, prog_bar=True)
    if pl_module.cfg.train.adversarial:
        pl_module.current_recon1 = recon1.detach()
        pl_module.current_recon2 = recon2.detach()
    return loss

def compute_loss(view1, pose1_hat, 
        silhouette1, recon1, view2,
        silhouette2, recon2, vertices, pl_module):

    if pl_module.cfg.train.adversarial:
        valid = torch.ones(recon2.size(0), 1).to(pl_module.device)
        critique1 = pl_module.discriminator(recon1)
        critique2 = pl_module.discriminator(recon2)
        critique = (critique1+critique2) / 2
        generator_loss = pl_module.gan_criterion(critique, valid)
    else:
        generator_loss = 0

    shape_loss = iou_loss(view1['alpha'], silhouette1) +\
            iou_loss(view2['alpha'], silhouette2)
    recon_loss1 = pl_module.recon_loss(view1['img'], recon1)
    recon_loss2 = pl_module.recon_loss(view2['img'],recon2)
    laplacian_loss = pl_module.laplacian_loss(vertices).mean()
    flatten_loss = pl_module.flatten_loss(vertices).mean()
    total_loss = (recon_loss1*pl_module.cfg.loss_weight.recon1
            + recon_loss2*pl_module.cfg.loss_weight.recon2
            + shape_loss*pl_module.cfg.loss_weight.shape
            + laplacian_loss*pl_module.cfg.loss_weight.laplacian
            + flatten_loss*pl_module.cfg.loss_weight.flatten
            + generator_loss*pl_module.cfg.loss_weight.generator)
    return total_loss

def discriminator_step(view1, view2, pl_module):
    valid = torch.ones(view1['img'].size(0), 1).to(pl_module.device)
    real_loss = pl_module.gan_criterion(
            pl_module.discriminator(view1['img']), valid)

    fake = torch.zeros(pl_module.current_recon2.size(0), 1).to(
            pl_module.device)
    fake_loss = (pl_module.gan_criterion(pl_module.discriminator(
                pl_module.current_recon1), fake) +
                pl_module.gan_criterion(pl_module.discriminator(
                pl_module.current_recon2), fake))/2

    discriminator_loss = (real_loss + fake_loss) / 2

    pl_module.log('discriminator_loss',
            discriminator_loss, prog_bar=True)
    return discriminator_loss

def validation(batch, batch_nb, pl_module):
    view1, view2 = batch
    #TODO: maybe do use predicted dist
    recon2 = pl_module.forward(view1, view2['pose'],
            novel_distance=torch.tensor(len(view1['img'])*[1.732]).to(pl_module.device))
    val_loss = pl_module.recon_loss(recon2, view2['img']) 
    return val_loss
