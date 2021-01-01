from torch import nn
import torch.nn.functional as F
import math
import torchvision

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
    def __init__(self, dim_in=3, img_size=128, latent_dim=1024):
        super(StandardEncoder, self).__init__()
        n_conv = int(math.log2(img_size) - 2)
        hidden_dims = [latent_dim//(2**i) for i in range(1,n_conv+1)[::-1]]
        self.conv_layers = [nn.Conv2d(dim_in, hidden_dims[0], kernel_size=5, stride=2, padding=2)]
        for i in range(n_conv-1):
            self.conv_layers.append(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1],
                        kernel_size=5, stride=2, padding=2))
        self.bn_layers = [nn.BatchNorm2d(dim) for dim in hidden_dims]

        self.fc1 = nn.Linear(hidden_dims[2]*math.ceil(img_size/8)**2,
                latent_dim*4)
        self.fc2 = nn.Linear(latent_dim*4, latent_dim*2)
        self.fc3 = nn.Linear(latent_dim*2, latent_dim)

    def forward(self, x):
        batch_size = len(x)
        out = x 
        for bn, cn in zip(self.bn_layers, self.conv_layers):
            out = F.relu(bn(cn(out)), inplace=True)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        return out

class VGGEncoder(nn.Module):
    def __init__(self, dim_in=3, img_size=128, dim_out=1000):
        super(VGGEncoder, self).__init__()
        self.vgg = torchvision.models.vgg16()

    def forward(self, x):
        out = self.vgg(x) 
        return out

class StandardDecoder(nn.Module):
    def __init__(self, dim_in=1024, dim_out=3, img_size=128):
        super(StandardDecoder, self).__init__()
        num_blks = int(math.log2(img_size))
        self.blks = [ResBlk(dim_in, dim_in//2, normalize=True, upsample=True,
                actv=nn.ReLU())]
        for i in range(1,num_blks):
            self.blks.append(ResBlk(dim_in//(2**i), dim_in//(2**(i+1)), normalize=True, upsample=True,
                    actv=nn.ReLU()))
        self.conv1x1 = nn.Conv2d(dim_in//img_size, 3,
                1, 1, 0)

    def forward(self, x):
        out = x.unsqueeze(-1).unsqueeze(-1)
        for blk in self.blks:
            out = blk(out)
        out = self.conv1x1(out)
        return out
