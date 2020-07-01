
import torch
from torch import nn
from torch.nn import functional as F
from codebase.nn import blocks, nets
from codebase.mathutils import log_prob_normal, repeat_newdim
from codebase.torchutils import *
from codebase.utils import *

import typing
import math
from PIL import Image
import os


hidden_dims_1 = [256, 256]
inter_dim = 64
hidden_dims_2 = [64, 64]



class HybridIWAE(nn.Module):

    def __str__(self):
        return "Hybrid_IWAE"

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                ):
        """
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.x_encoder = nets.LinearMap(X_DIM, [200], 50)
        self.z_encoder_xz = nets.LinearMap(Z_DIM, [200], 50)
        self.z_encoder_zv = nets.LinearMap(Z_DIM, [200], 50)
        self.v_encoder = nets.LinearMap(V_DIM, [200], 50)

        # inference networks

        xz_reduced_dim = self.x_encoder.get_output_dim() + self.z_encoder_xz.get_output_dim()
        self.v_given_xz = nets.GaussianStochasticEncoder(xz_reduced_dim, V_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)

        zv_reduced_dim = self.z_encoder_zv.get_output_dim() + self.v_encoder.get_output_dim()
        self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU(), output_act=None)


        
    def forward(self,
                x, 
                y,
                z_given_x,
                n_samples=1,
                block_grad=False,
                ):
        """
        z_given_x inference network is trained separately with elbo objective
        """

        
        # for k particles of z, repeat x k times
        x = repeat_newdim(x, n_samples, -2)
        y = repeat_newdim(y, n_samples, -2)
        z, _, _, log_posterior_z_given_x = z_given_x(x)


        x_reduced = self.x_encoder(x)
        z_reduced_xz = self.z_encoder_xz(z)
        xz_reduced = torch.cat((x_reduced, z_reduced_xz), dim=-1)
        v, _, _, log_posterior_v_given_xz = self.v_given_xz(xz_reduced)

        log_prior_z = log_normal_prior(z)
        log_prior_v = log_normal_prior(v)

        z_reduced_zv = self.z_encoder_zv(z)
        v_reduced = self.v_encoder(v)        
        zv_reduced = torch.cat((z_reduced_zv, v_reduced), dim=-1)
        x_recon = self.x_given_zv(zv_reduced)
        y_recon = self.y_given_z(z)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles


        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xz.squeeze_(dim=-1)
        log_posterior_z_given_x.squeeze_(dim=-1)



        w = img_log_likelihood_x + img_log_likelihood_y + log_prior_z + log_prior_v \
            - log_posterior_z_given_x - log_posterior_v_given_xz

        # elbo
        if w.shape[-1] != 1:
            elbo = w.mean(dim=-1, keepdim=True)
            iw_elbo = log_mean_exp(w, dim=-1, detach=True)

        elbo.squeeze_(-1)
        iw_elbo.squeeze_(-1)

        return iw_elbo, elbo  

