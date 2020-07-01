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



class PartIwaePretrainedXY(nn.Module):

    def __str__(self):
        return "Partial_IWAE"

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                ):
        """
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.x_encoder = nets.LinearMap(X_DIM, [500], 50)
        self.z_encoder_xz = nets.LinearMap(Z_DIM, [500], 50)
        self.z_encoder_zv = nets.LinearMap(Z_DIM, [500], 50)
        self.v_encoder = nets.LinearMap(V_DIM, [500], 50)

        # inference networks
        self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU())

        xz_reduced_dim = self.x_encoder.get_output_dim() + self.z_encoder_xz.get_output_dim()
        self.v_given_xz = nets.GaussianStochasticEncoder(xz_reduced_dim, V_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU(), output_act=None)

        zv_reduced_dim = self.z_encoder_zv.get_output_dim() + self.v_encoder.get_output_dim()
        self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[500, 500], hidden_acts=nn.ReLU(), output_act=None)


        
    def forward(self,
                x, 
                y,
                n_samples_z=1,
                n_samples_v=1,
                inner_method='iwae',
                outer_method='elbo',
                block_grad=False,
                ):
        """
        For Full training Method (ELBO, IWAE), n_samples_v should be 1, and inner_method doesnt matter. outer_method decides the type of bound

        For repeating random samples, I repeat the input data. I don't know if this is faster
        However, I need to repeat input data to compute likelihood regardless

        What to repeat in the inner bound: v,z and x
        """
        
        # for k particles of z, repeat x k times
        x_outer = repeat_newdim(x, n_samples_z, -2)
        z_outer, _, _, log_posterior_z_given_x = self.z_given_x(x_outer)

        # for n_samples_v particles of v, repeat z n_samples_v times, x_re k times and y n_samples_z * n_samples_v times
        #TODO: very akward, better way?
        y_outer = repeat_newdim(y, n_samples_z, -2)
        x_inner = repeat_newdim(x_outer, n_samples_v, -2)
        z_inner = repeat_newdim(z_outer, n_samples_v, -2)

        ##############################
        # tentative
        if block_grad:
            z_inner = z_inner.detach()
            assert(z_inner.requires_grad is False)

        # thus, for each (x,y) pair, there are {n_samples_z * n_samples_v} v particles
        # there are z_outer and z_inner for the fact that log_q(z_outer|x) in the outer sum can be used right away
        x_reduced = self.x_encoder(x_inner)
        z_reduced_xz = self.z_encoder_xz(z_inner)
        xz_reduced = torch.cat((x_reduced, z_reduced_xz), dim=-1)
        v_inner, _, _, log_posterior_v_given_xz = self.v_given_xz(xz_reduced)

        log_prior_z = log_normal_prior(z_outer)
        log_prior_v = log_normal_prior(v_inner)

        z_reduced_zv = self.z_encoder_zv(z_inner)
        v_reduced = self.v_encoder(v_inner)        
        zv_reduced = torch.cat((z_reduced_zv, v_reduced), dim=-1)
        x_recon = self.x_given_zv(zv_reduced)
        y_recon = self.y_given_z(z_outer)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x_inner)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y_outer)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles

        # compute iwae q(v|x,y,z) | p(x,y,z,v)
        # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xz.squeeze_(dim=-1)
        log_posterior_z_given_x.squeeze_(dim=-1)




        inner_lowerbound = img_log_likelihood_x + log_prior_v - log_posterior_v_given_xz
        # inner_lowerbound = img_log_likelihood_x + self.reg * img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_v_given_xz
        if inner_lowerbound.shape[-1] != 1:
            if inner_method == 'iwae':
                inner_lowerbound = log_mean_exp(inner_lowerbound, dim=-1, detach=True)
                inner_lowerbound = inner_lowerbound.sum(dim=-1, keepdim=True)
            elif inner_method == 'elbo':
                inner_lowerbound = inner_lowerbound.mean(dim=-1, keepdim=True)
        inner_lowerbound.squeeze_(-1)

        log_p_x_lowerbound = inner_lowerbound + log_prior_z - log_posterior_z_given_x
        outer_lowerbound = log_p_x_lowerbound + img_log_likelihood_y
        # outer_lowerbound = inner_lowerbound - log_posterior_z_given_x
        if outer_lowerbound.shape[-1] != 1:
            if outer_method== 'iwae':
                outer_lowerbound = log_mean_exp(outer_lowerbound, dim=-1, detach=True)
                outer_lowerbound = outer_lowerbound.sum(dim=-1, keepdim=True)
            elif outer_method == 'elbo':
                outer_lowerbound = outer_lowerbound.mean(dim=-1, keepdim=True)
        outer_lowerbound.squeeze_(-1)



        return outer_lowerbound


# class FullELBO(nn.Module):
#     def __str__(self):
#         return "Full_ELBO"

#     def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
#                 ):
#         """
#         #TODO: need to move batch enlargement in here.

#         :arg reg: increase weight of \log p(y|z). This option is tentative
#         """
#         super().__init__()
#         self.X_DIM = X_DIM
#         self.Y_DIM = Y_DIM
#         self.Z_DIM = Z_DIM
#         self.V_DIM = V_DIM
#         self.x_encoder = nets.LinearMap(X_DIM, [200, 200], 50)
#         self.z_encoder_xz = nets.LinearMap(Z_DIM, [200, 200], 50)
#         self.z_encoder_zv = nets.LinearMap(Z_DIM, [200, 200], 50)
#         self.v_encoder = nets.LinearMap(V_DIM, [200, 200], 50)

#         # inference networks
#         self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

#         xz_reduced_dim = self.x_encoder.get_output_dim() + self.z_encoder_xz.get_output_dim()
#         self.v_given_xz = nets.GaussianStochasticEncoder(xz_reduced_dim, V_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

#         # generative networks
#         self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)

#         zv_reduced_dim = self.z_encoder_zv.get_output_dim() + self.v_encoder.get_output_dim()
#         self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)


        
#     def forward(self,
#                 x, 
#                 y,
#                 n_samples=1,
#                 method='elbo',
#                 ):
#         """
#         first estimator: $VAE\left(q(z|x), IWAE(q(v|x,y,z),p(x,y,z,v))\right)$

#         For repeating random samples, I repeat the input data. I don't know if this is faster
#         However, I need to repeat input data to compute likelihood regardless

#         What to repeat in the inner bound: v,z and x
#         """
        
#         # for k particles of z, repeat x k times
#         x = repeat_newdim(x, n_samples, -2)
#         y = repeat_newdim(y, n_samples, -2)
#         z, _, _, log_posterior_z_given_x = self.z_given_x(x)

#         # sample v from q(v|z,x)
#         x_reduced = self.x_encoder(x)
#         z_reduced_xz = self.z_encoder_xz(z)
#         xz_reduced = torch.cat((x_reduced, z_reduced_xz), dim=-1)
#         v, _, _, log_posterior_v_given_xz = self.v_given_xz(xz_reduced)

#         log_prior_z = log_normal_prior(z)
#         log_prior_v = log_normal_prior(v)

#         z_reduced_zv = self.z_encoder_zv(z)
#         v_reduced = self.v_encoder(v)        
#         zv_reduced = torch.cat((z_reduced_zv, v_reduced), dim=-1)
#         x_recon = self.x_given_zv(zv_reduced)
#         y_recon = self.y_given_z(z)

#         img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
#         img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
#         # for iwae, average over particles
#         # for elbo, sum over weighted particles

#         # compute iwae q(v|x,y,z) | p(x,y,z,v)
#         # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

#         img_log_likelihood_x.squeeze_(dim=-1)
#         img_log_likelihood_y.squeeze_(dim=-1)
#         log_prior_z.squeeze_(dim=-1)
#         log_prior_v.squeeze_(dim=-1)
#         log_posterior_v_given_xz.squeeze_(dim=-1)
#         log_posterior_z_given_x.squeeze_(dim=-1)


#         lowerbound = img_log_likelihood_x + img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_z_given_x - log_posterior_v_given_xz 


#         if method == 'iwae':

#             lowerbound = get_importance_bound(lowerbound)
#             lowerbound = lowerbound.sum(dim=-1, keepdim=True)

#         if method == 'elbo':
#             lowerbound = lowerbound.mean(dim=-1, keepdim=True)

#         lowerbound = lowerbound.squeeze_(dim=-1)
#         return lowerbound