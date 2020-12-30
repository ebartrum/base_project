from torch import nn
import torch.nn.functional as F
import math

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StandardEncoder(nn.Module):
    def __init__(self, dim_in=3, img_size=128):
        super(StandardEncoder, self).__init__()

        latent_dim = 1024
        hidden_dims = [latent_dim//8, latent_dim//4, latent_dim//2]

        self.conv1 = nn.Conv2d(dim_in, hidden_dims[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])

        self.fc1 = nn.Linear(hidden_dims[2]*math.ceil(img_size/8)**2,
                latent_dim*4)
        self.fc2 = nn.Linear(latent_dim*4, latent_dim*2)
        self.fc3 = nn.Linear(latent_dim*2, latent_dim)

    def forward(self, x):
        batch_size = len(x)
        out = x 
        out = F.relu(self.bn1(self.conv1(out)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        return out

# class StandardDecoder(nn.Module):
#     def __init__(self):
#         super(StandardDecoder, self).__init__()
#         latent_dim = 1024
#         hidden_dims = [latent_dim, latent_dim//2, latent_dim//4, latent_dim//8]
#         self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
#         self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])

#     def forward(self, x):
#         batch_size = len(x)
#         x = F.relu(self.fc1(x), inplace=True)
#         x = F.relu(self.fc2(x), inplace=True)

#         return out

class StandardDecoder(nn.Module):
    def __init__(self, dim_in=1024, dim_out=3):
        super(StandardDecoder, self).__init__()
        self.blk1 = ResBlk(dim_in, dim_in//2, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk2 = ResBlk(dim_in//2, dim_in//4, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk3 = ResBlk(dim_in//4, dim_in//8, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk4 = ResBlk(dim_in//8, dim_in//16, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk5 = ResBlk(dim_in//16, dim_in//32, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk6 = ResBlk(dim_in//32, dim_in//64, normalize=True, upsample=True,
                actv=nn.ReLU())
        self.blk7 = ResBlk(dim_in//64, dim_in//128, normalize=True, upsample=True,
                actv=nn.ReLU())

        dim_hidden = [dim_in//2, dim_in//4]
        self.conv1x1 = nn.Conv2d(dim_in//128, 3,
                1, 1, 0)

    def forward(self, x):
        out = x.unsqueeze(-1).unsqueeze(-1)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = self.blk5(out)
        out = self.blk6(out)
        out = self.blk7(out)

        out = self.conv1x1(out)
        return out
