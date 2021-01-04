from abc import ABC, abstractmethod
import os
import numpy as np
import imageio
import torch
import torchvision
from torch.utils.data import DataLoader
import hydra

class Figure(ABC):
    def __init__(self, cfg):
       self.save_dir = cfg.dir 
       self.filename = cfg.filename if cfg.filename else\
               f"{self.__class__.__name__}.png"
       if not os.path.exists(self.save_dir):
           os.makedirs(self.save_dir)

    @abstractmethod
    def draw(self, pl_module):
        """
        Draw figure as a numpy array. Type should be float or double.
        Range should be in [0,1]. Dim should be (H,W,3)
        """
        pass

    def save(self, array):
        assert array.min()>=0 and array.max()<=1,\
                "Figure array should lie in [0,1]"
        array = (array*255).astype(int)
        imageio.imwrite(f"{self.save_dir}/{self.filename}", array)

    def draw_and_save(self, pl_module):
        fig_array = self.draw(pl_module)
        self.save(fig_array)

class RainbowSquare(Figure):
    def __init__(self, cfg):
        super(RainbowSquare, self).__init__(cfg)

    def draw(self, pl_module):
        fig_array = np.random.random((512,512,3))
        return fig_array

class ReconGrid(Figure):
    def __init__(self, cfg, num_imgs):
        super(ReconGrid, self).__init__(cfg)
        self.dataset = hydra.utils.instantiate(cfg.datasets.val)
        self.num_imgs = num_imgs
        self.dataloader = DataLoader(self.dataset,
                batch_size=self.num_imgs,shuffle=False)
        self.input_imgs = self.dataloader.collate_fn(
                [self.dataset[i] for i in range(self.num_imgs)])

    @torch.no_grad()
    def draw(self, pl_module):
        img = self.input_imgs.to(pl_module.device)
        recon = pl_module.forward(img)
        grid = torchvision.utils.make_grid(torch.cat(
            [img,recon],dim=0),
            nrow=self.num_imgs)
        grid = grid.permute(1,2,0)
        grid = torch.clamp(grid, 0, 1)
        fig_array = grid.detach().cpu().numpy()
        return fig_array
