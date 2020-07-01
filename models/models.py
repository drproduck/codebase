import torch
from torch import nn
import torch.nn.functional as F
import pdb
import sys
# from codebase.nn.nets import StochasticEncoder, StochasticDecoder
from codebase.mathutils import log_prob_normal, log_mean_exp

import typing
from codebase.nn import blocks

class VAE(nn.Module):
    """
    VAE with unit gaussian prior and gaussian variational posterior
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def log_likelihood(self, x, z):
        logits = self.decoder(z)
        pixel_log_likelihood = -1 * torch.binary_cross_entropy_with_logits(logits, x, reduction='none')
        return pixel_log_likelihood.sum(dim=-1)

    def log_prior(self, z):
        return log_prob_normal(z, 0)

    def log_posterior(self, z, mu, var):
        return log_prob_normal(z, mu, var)

    def nll_lowerbound(self, x_batch, method='elbo', n_samples=5000):
        nll = 0.0
        x_batch = x_batch.unsqueeze(1).repeat(1, n_samples, 1)
        z, mu, var = self.encoder(x_batch)
        log_prior = self.log_prior(z)
        log_likelihood = self.log_likelihood(x_batch, z)
        log_posterior = self.log_posterior(z, mu, var)
        log_weight = log_likelihood + log_prior - log_posterior

        # average over z particles, sum over x samples. The outer function will divide
        if method == 'elbo' or method is None:
            nll = nll - log_weight.mean(-1).sum()

        if method == 'iwae':
            nll = nll - log_mean_exp(log_weight).sum()
            # pdb.set_trace()
                

        return nll


class ImageNet(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.map = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, stride=1),
                                nn.BatchNorm2d(num_features=32),
                                nn.ELU(),
                                nn.Conv2d(in_channels=32, out_channels=64, stride=2),
                                nn.BatchNorm2d(num_features=32),
                                nn.ELU(),
                                nn.Conv2d(in_channels=64, out_channels=128, stride=2),
                                nn.BatchNorm2d(num_features=32),
                                nn.ELU(),
                                nn.Conv2d(in_channels=128, out_channels=16, stride=2),
                                )
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.map(x)
        x = x.view(batch_size, -1)
        return x




class JointVAEEncoder(nn.Module):
    def __init__(self, class_dim, scale_dim, orientation_dim, location_dim, img_dim, z_dim):
        super().__init__()
        self.class_dim = class_dim
        self.scale_dim = scale_dim
        self.orientation_dim = orientation_dim
        self.location_dim = location_dim
        self.img_dim = img_dim
        self.z_dim = z_dim

        self.class_net = nn.Sequential(nn.Linear(self.class_dim, 32), nn.Linear(32, 512), nn.Linear(512, 512))
        self.scale_net = nn.Sequential(nn.Linear(self.scale_dim, 32), nn.Linear(32, 512), nn.Linear(512, 512))
        self.orientation_net = nn.Sequential(nn.Linear(self.orientation_dim, 32), nn.Linear(32, 512), nn.Linear(512, 512))
        self.location_net = nn.Sequential(nn.Linear(self.location_dim, 32), nn.Linear(32, 512), nn.Linear(512, 512))

        self.label_cat_net = nn.Sequential(nn.Linear(2048, 512), nn,Linear(512, 512))

        self.img_net = ImageNet(self.img_dim)

        self.cat_net = Encoder(hidden_dims=[512, 512], x_dim=1536, z_dim=self.z_dim)


    def forward(self, img, clss, scale, orientation, location):
        clss = self.class_net(clss)
        scale = self.scale_net(scale)
        orientation = self.orientation_net(orientation)
        location = self.location_net(location)

        labels = torch.cat((clss, scale, orientation, location), dim=-1)
        labels = self.label_cat_net(labels)

        img = self.img_net(img)

        feature = torch.cat((labels, img), dim=-1)
        z, mu, var = self.cat_net(feature)

        return z, mu, var



class JointVAEImageDecoder(nn.Module):
    def __init__(self, z_dim, feature_dim, out_channels):
        super().__init__()
        #TODO: figure out how to change dimension of last map for image of other sizes
        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.map = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
            # state size. (feature_dim*8) x 4 x 4
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            # state size. (feature_dim*4) x 8 x 8
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            # state size. (feature_dim*2) x 16 x 16
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            # state size. (feature_dim) x 32 x 32
            nn.ConvTranspose2d(feature_dim, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        bernoullis = self.map(z)
        return bernoullis

class JointVAELabelDecoder(nn.Module):
    def __init__(self, z_dim, clss, scale_dim, orientation_dim, location_dim):
        super().__init__()
        self.class_decoder = nn.Sequential(nn.Linear(z_dim, 128), nn.Linear(128, 128), nn.Linear(128, class_dim), nn.Tanh())
        self.scale_decoder = nn.Sequential(nn.Linear(z_dim, 128), nn.Linear(128, 128), nn.Linear(128, scale_dim), nn.Tanh())
        self.orientation_decoder = nn.Sequential(nn.Linear(z_dim, 128), nn.Linear(128, 128), nn.Linear(128, orientation_dim), nn.Tanh())
        self.location_decoder = nn.Sequential(nn.Linear(z_dim, 128), nn.Linear(128, 128), nn.Linear(128, location_dim), nn.Tanh())

    def forward(self, z):
        clss = self.class_decoder(z)
        scale = self.scale_decoder(z)
        orientation = self.orientation_decoder(z)
        location = self.location_decoder(z)
        return clss, scale, orientation, location

class JointVAE(nn.Module):
    def __init__(self, class_dim, scale_dim, orientation_dim, location_dim, img_dim, z_dim, feature_dim, out_channels):
        super().__init__()
        self.encoder = JointVAEEncoder(class_dim, scale_dim, orientation_dim, location_dim, img_dim, z_dim)
        self.img_decoder = JointVAEImageDecoder(z_dim, feature_dim, out_channels)
        self.label_decoder = JointVAELabelDecoder(z_dim, class_dim, scale_dim, orientation_dim, location_dim)


    def log_prior(self, z):
        return log_prob_normal(z, 0.0)

    def log_posterior(self, z, mu, var):
         return log_prob_normal(z, mu, var)
        

    def forward(self, img, clss, scale, orientation, location):
        z, mu, var = self.encoder(img, clss, scale, orientation, location)
        img_recon = self.decoder(z)
        class_recon, scale_recon, orientation_recon, location_recon = self.decoder(z)

        return img_recon, class_recon, scale_recon, orientation_recon, location_recon


        

