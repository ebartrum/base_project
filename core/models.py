from torch import nn

class StandardEncoder(nn.Module):
    def __init__(self, dim_in=3):
        super(StandardEncoder, self).__init__()

        latent_dim = 1024
        hidden_dims = [latent_dim//8, latent_dim//4, latent_dim//2]

        self.conv1 = nn.Conv2d(dim_in, hidden_dims[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])

        self.fc = nn.Linear(hidden_dims[2], latent_dim)

    def forward(self, x):
        import ipdb;ipdb.set_trace()
        batch_size = len(x)
        out = x 
        out = F.relu(self.bn1(self.conv1(out)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)), inplace=True)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out), inplace=True)
        return out

class StandardDecoder(nn.Module):
    def __init__(self):
        super(StandardDecoder, self).__init__()
        latent_dim = 1024
        hidden_dims = [latent_dim, latent_dim//2, latent_dim//4, latent_dim//8]
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[2])

    def forward(self, x):
        batch_size = len(x)
        import ipdb;ipdb.set_trace()
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)


        return out
