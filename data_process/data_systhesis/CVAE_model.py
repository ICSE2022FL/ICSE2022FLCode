import math

import torch
from torch import nn
from torch.nn import functional as F


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    idx = idx.squeeze(0).squeeze(0)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot.unsqueeze(0).unsqueeze(0)


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.mid_channels = 15
        self.out_channels = 30
        self.kernel_size = 4
        self.step = 2
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.mid_channels,
                               kernel_size=(1, self.kernel_size))

        self.conv2 = nn.Conv2d(in_channels=self.mid_channels,
                               out_channels=self.out_channels,
                               kernel_size=(1, self.kernel_size))

        self.hidden_units = math.floor((layer_sizes[0] - self.kernel_size + 1) / self.step)
        self.hidden_units = self.out_channels * (math.floor((self.hidden_units - self.kernel_size + 1) / self.step))

        self.linear_means = nn.Linear(self.hidden_units, latent_size)
        self.linear_log_var = nn.Linear(self.hidden_units, latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=2)
            x = torch.cat((x, c), dim=-1)

        h1 = F.max_pool2d(F.relu(self.conv1(x)), (1, self.step))
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), (1, self.step))
        h2 = h2.view(-1, self.hidden_units)

        means = self.linear_means(h2)
        log_vars = self.linear_log_var(h2)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        self.linear = nn.Linear(in_features=input_size, out_features=layer_sizes[0])

        self.mid_channels = 15
        self.out_channels = 30
        self.kernel_size = 2
        self.step = 1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.mid_channels,
                               kernel_size=(1, self.kernel_size))

        self.conv2 = nn.Conv2d(in_channels=self.mid_channels,
                               out_channels=self.out_channels,
                               kernel_size=(1, self.kernel_size))

        self.hidden_units = math.floor((layer_sizes[0] - self.kernel_size + 1) / self.step)
        self.hidden_units = self.out_channels * (math.floor((self.hidden_units - self.kernel_size + 1) / self.step))

        self.out = nn.Linear(self.hidden_units, layer_sizes[1])

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=2).squeeze(0).squeeze(0)
            z = torch.cat((z, c), dim=-1)

        h = self.linear(z).unsqueeze(0).unsqueeze(0)
        h1 = F.max_pool2d(F.relu(self.conv1(h)), (1, self.step))
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), (1, self.step))
        h2 = h2.view(-1, self.hidden_units)
        x = self.out(h2)
        return x


class CVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):
        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


def loss_fn(recon_x, x, mean, log_var):
    MSE = (recon_x - x).norm(2).pow(2)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (MSE + KLD) / x.size(0)
