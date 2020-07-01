import torch
from torch import nn
from torch.nn import functional as F
from codebase.models import VAE
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


#TODO: have 2 different encoders for z->xyz and z->zv, or just 1???


class PIWAE(object):
    def __init__(self):
        pass

class ELBO_1(nn.Module, PIWAE):

    def __str__(self):
        return "ELBO 1st bound"

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                reg=1.0,
                ):
        """
        #TODO: need to move batch enlargement in here.

        :arg reg: increase weight of \log p(y|z). This option is tentative
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.y_encoder = nets.LinearMap(Y_DIM, [200, 200], 50)
        self.x_encoder = nets.LinearMap(X_DIM, [200, 200], 50)
        self.z_encoder = nets.LinearMap(Z_DIM, [200, 200], 50)
        self.v_encoder = nets.LinearMap(V_DIM, [200, 200], 50)

        # tentative 
        self.reg = reg

        # inference networks
        self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())
        xyz_reduced_dim = self.x_encoder.get_output_dim() + self.y_encoder.get_output_dim() + self.z_encoder.get_output_dim()
        self.v_given_xyz = nets.GaussianStochasticEncoder(xyz_reduced_dim, V_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)
        zv_reduced_dim = self.z_encoder.get_output_dim() + self.v_encoder.get_output_dim()

        self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)


        
    def forward(self,
                x, 
                y,
                n_samples_z=1,
                n_samples_v=1,
                inner_method='iwae',
                outer_method='elbo',
                ):
        """
        first estimator: $VAE\left(q(z|x), IWAE(q(v|x,y,z),p(x,y,z,v))\right)$


        For repeating random samples, I repeat the input data. I don't know if this is faster
        However, I need to repeat input data to compute likelihood regardless
        """
        
        # for k particles of z, repeat x k times
        x = repeat_newdim(x, n_samples_z, -2)

        z_outer, mu_z_given_x, var_z_given_x, log_posterior_z_given_x = self.z_given_x(x)

        # for n_samples_v particles of v, repeat z n_samples_v times, x_re k times and y n_samples_z * n_samples_v times
        #TODO: very akward, better way?
        x = repeat_newdim(x, n_samples_v, -2)
        z_inner = repeat_newdim(z_outer, n_samples_v, -2)
        y = repeat_newdim(y, n_samples_z, -2)
        y = repeat_newdim(y, n_samples_v, -2)

        # thus, for each (x,y) pair, there are {n_samples_z * n_samples_v} v particles
        # there are z_outer and z_inner for the fact that log_q(z_outer|x) in the outer sum can be used right away
        x_reduced = self.x_encoder(x)
        y_reduced = self.y_encoder(y)
        z_reduced = self.z_encoder(z_inner)
        xyz_reduced = torch.cat((x_reduced, y_reduced, z_reduced), dim=-1)
        v, mu, var, log_posterior_v_given_xyz = self.v_given_xyz(xyz_reduced)
        log_prior_z = log_normal_prior(z_inner)
        log_prior_v = log_normal_prior(v)

        v_reduced = self.v_encoder(v)        
        zv_reduced = torch.cat((z_reduced, v_reduced), dim=-1)
        x_recon = self.x_given_zv(zv_reduced)
        y_recon = self.y_given_z(z_inner)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles

        # compute iwae q(v|x,y,z) | p(x,y,z,v)
        # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xyz.squeeze_(dim=-1)
        log_posterior_z_given_x.squeeze_(dim=-1)

        inner_lowerbound = img_log_likelihood_x + self.reg * img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_v_given_xyz
        if inner_method == 'iwae':
            inner_lowerbound = get_importance_bound(inner_lowerbound)
            inner_lowerbound = inner_lowerbound.sum(dim=-1, keepdim=True)
        elif inner_method == 'elbo':
            inner_lowerbound = inner_lowerbound.mean(dim=-1, keepdim=True)
        inner_lowerbound.squeeze_(-1)

        outer_lowerbound = inner_lowerbound - log_posterior_z_given_x
        if outer_method== 'iwae':
            outer_lowerbound = get_importance_bound(outer_lowerbound)
            outer_lowerbound = outer_lowerbound.sum(dim=-1, keepdim=True)
        elif outer_method == 'elbo':
            outer_lowerbound = outer_lowerbound.mean(dim=-1, keepdim=True)
        outer_lowerbound.squeeze_(-1)


        return outer_lowerbound


class ELBO_2(nn.Module, PIWAE):
    def __str__(self):
        return 'ELBO 2nd bound'

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                ):
        """
        second estimator: $VAE\left(q(z|x)l(z;y), IWAE(q(v|x,y,z),p(x,y,z,v))\right)$
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.y_encoder = nets.LinearMap(Y_DIM, [200, 200], 50)
        self.x_encoder = nets.LinearMap(X_DIM, [200, 200], 50)
        self.z_encoder = nets.LinearMap(Z_DIM, [200, 200], 50)
        self.v_encoder = nets.LinearMap(V_DIM, [200, 200], 50)

        # inference networks
        self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())
            # factored inference, approximating p(z|x,y) = q(z|x)q(z|y)
        self.z_given_y = nets.GaussianStochasticEncoder(Y_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        xyz_reduced_dim = self.x_encoder.get_output_dim() + self.y_encoder.get_output_dim() + self.z_encoder.get_output_dim()
        self.v_given_xyz = nets.GaussianStochasticEncoder(xyz_reduced_dim, V_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)
        zv_reduced_dim = self.z_encoder.get_output_dim() + self.v_encoder.get_output_dim()

        self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)


    def forward(self,
                x, 
                y,
                n_samples_z=1,
                n_samples_v=1,
                inner_method='iwae',
                outer_method='elbo',
                ):
        """

        For repeating random samples, I repeat the input data. I don't know if this is faster
        However, I need to repeat input data to compute likelihood regardless
        """
        
        # for k particles of z, repeat x, y k times
        x = repeat_newdim(x, n_samples_z, -2)
        y = repeat_newdim(y, n_samples_z, -2)

        _, mu_z_given_x, var_z_given_x, _ = self.z_given_x(x)
        _, mu_z_given_y, var_z_given_y, _ = self.z_given_y(y)
        mu_z_given_xy, var_z_given_xy = product_of_diag_normals([mu_z_given_x, mu_z_given_y], 
                                                                [var_z_given_x, var_z_given_y])

        z_outer = mu_z_given_xy + torch.Tensor(mu_z_given_xy.shape).normal_().cuda() * var_z_given_xy.sqrt()
        log_posterior_z_given_xy = log_prob_normal(z_outer, mu_z_given_xy, var_z_given_xy)

        # for n_samples_v particles of v, repeat z_outer, n_samples_v times, x, y another n_samples_v times
        #TODO: very akward, better way?
        z_inner = repeat_newdim(z_outer, n_samples_v, -2)
        x = repeat_newdim(x, n_samples_v, -2)
        y = repeat_newdim(y, n_samples_v, -2)

        # thus, for each (x,y) pair, there are {n_samples_z * n_samples_v} v particles
        # there are z_outer and z_inner for the fact that log_q(z_outer|x,y) in the outer sum can be used right away

        #TODO: have 2 different encoders for z->xyz and z->zv, or just 1???
        #TODO: or even have no encoders for z AND v at all?
        x_reduced = self.x_encoder(x)
        y_reduced = self.y_encoder(y)
        z_reduced = self.z_encoder(z_inner)
        xyz_reduced = torch.cat((x_reduced, y_reduced, z_reduced), dim=-1)
        v, mu, var, log_posterior_v_given_xyz = self.v_given_xyz(xyz_reduced)
        log_prior_z = log_normal_prior(z_inner)
        log_prior_v = log_normal_prior(v)

        v_reduced = self.v_encoder(v)
        zv_reduced = torch.cat((z_reduced, v_reduced), dim=-1)
        x_recon = self.x_given_zv(zv_reduced)
        y_recon = self.y_given_z(z_inner)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles

        # compute iwae q(v|x,y,z) | p(x,y,z,v)
        # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xyz.squeeze_(dim=-1)
        log_posterior_z_given_xy.squeeze_(dim=-1)

        inner_lowerbound = img_log_likelihood_x + img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_v_given_xyz
        if inner_method == 'iwae':
            inner_lowerbound = get_importance_bound(inner_lowerbound)
            inner_lowerbound = inner_lowerbound.sum(dim=-1, keepdim=True)
        elif inner_method == 'elbo':
            inner_lowerbound = inner_lowerbound.mean(dim=-1, keepdim=True)
        inner_lowerbound.squeeze_(-1)

        outer_lowerbound = inner_lowerbound - log_posterior_z_given_xy
        if outer_method== 'iwae':
            outer_lowerbound = get_importance_bound(outer_lowerbound)
            outer_lowerbound = outer_lowerbound.sum(dim=-1, keepdim=True)
        elif outer_method == 'elbo':
            outer_lowerbound = outer_lowerbound.mean(dim=-1, keepdim=True)
        outer_lowerbound.squeeze_(-1)


        return outer_lowerbound


class ELBO_2_ECON(nn.Module, PIWAE):
    def __str__(self):
        return 'ELBO 2nd bound with economic parameterization'

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                ):
        """
        second estimator: $VAE\left(q(z|x)l(z;y), IWAE(q(v|x,y,z),p(x,y,z,v))\right)$

        NO encoders for z and v
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.y_encoder = nets.LinearMap(Y_DIM, [200, 200], 50)
        self.x_encoder = nets.LinearMap(X_DIM, [200, 200], 50)

        # inference networks
        self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())
            # factored inference, approximating p(z|x,y) = q(z|x)q(z|y)
        self.z_given_y = nets.GaussianStochasticEncoder(Y_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        xyz_reduced_dim = self.x_encoder.get_output_dim() + self.y_encoder.get_output_dim() + self.Z_DIM
        self.v_given_xyz = nets.GaussianStochasticEncoder(xyz_reduced_dim, V_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)

        zv_dim = self.Z_DIM + self.V_DIM
        self.x_given_zv = nets.StochasticDecoder(zv_dim, X_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)


    def forward(self,
                x, 
                y,
                n_samples_z=1,
                n_samples_v=1,
                inner_method='iwae',
                outer_method='elbo',
                ):
        """
        """
        
        # for k particles of z, repeat x, y k times
        x = repeat_newdim(x, n_samples_z, -2)
        y = repeat_newdim(y, n_samples_z, -2)

        _, mu_z_given_x, var_z_given_x, log_posterior_z_given_x = self.z_given_x(x)
        _, mu_z_given_y, var_z_given_y, log_posterior_z_given_y = self.z_given_y(y)
        mu_z_given_xy, var_z_given_xy = product_of_diag_normals([mu_z_given_x, mu_z_given_y], 
                                                                [var_z_given_x, var_z_given_y])

        z_outer = mu_z_given_xy + torch.Tensor(mu_z_given_xy.shape).normal_().cuda() * var_z_given_xy.sqrt()
        log_posterior_z_given_xy = log_prob_normal(z_outer, mu_z_given_xy, var_z_given_xy)

        # for n_samples_v particles of v, repeat z_outer, n_samples_v times, x, y another n_samples_v times
        #TODO: very akward, better way?
        z_inner = repeat_newdim(z_outer, n_samples_v, -2)
        x = repeat_newdim(x, n_samples_v, -2)
        y = repeat_newdim(y, n_samples_v, -2)

        # thus, for each (x,y) pair, there are {n_samples_z * n_samples_v} v particles
        # there are z_outer and z_inner for the fact that log_q(z_outer|x,y) in the outer sum can be used right away

        x_reduced = self.x_encoder(x)
        y_reduced = self.y_encoder(y)

        xyz_reduced = torch.cat((x_reduced, y_reduced, z_inner), dim=-1)
        v, mu, var, log_posterior_v_given_xyz = self.v_given_xyz(xyz_reduced)
        log_prior_z = log_normal_prior(z_inner)
        log_prior_v = log_normal_prior(v)

        zv = torch.cat((z_inner, v), dim=-1)
        x_recon = self.x_given_zv(zv)
        y_recon = self.y_given_z(z_inner)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles

        # compute iwae q(v|x,y,z) | p(x,y,z,v)
        # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xyz.squeeze_(dim=-1)
        log_posterior_z_given_xy.squeeze_(dim=-1)

        inner_lowerbound = img_log_likelihood_x + img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_v_given_xyz
        if inner_method == 'iwae':
            inner_lowerbound = get_importance_bound(inner_lowerbound)
            inner_lowerbound = inner_lowerbound.sum(dim=-1, keepdim=True)
        elif inner_method == 'elbo':
            inner_lowerbound = inner_lowerbound.mean(dim=-1, keepdim=True)
        inner_lowerbound.squeeze_(-1)

        outer_lowerbound = inner_lowerbound - log_posterior_z_given_xy
        if outer_method== 'iwae':
            outer_lowerbound = get_importance_bound(outer_lowerbound)
            outer_lowerbound = outer_lowerbound.sum(dim=-1, keepdim=True)
        elif outer_method == 'elbo':
            outer_lowerbound = outer_lowerbound.mean(dim=-1, keepdim=True)
        outer_lowerbound.squeeze_(-1)


        return outer_lowerbound
    
class ELBO_1_ECON(nn.Module, PIWAE):

    def __str__(self):
        return "ELBO 1st bound with economic parameterization"

    def __init__(self, X_DIM, Y_DIM, Z_DIM, V_DIM,
                ):
        """
        #TODO: need to move batch enlargement in here.
        """
        super().__init__()
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.Z_DIM = Z_DIM
        self.V_DIM = V_DIM
        self.y_encoder = nets.LinearMap(Y_DIM, [200, 200], 50)
        self.x_encoder = nets.LinearMap(X_DIM, [200, 200], 50)

        # inference networks
        self.z_given_x = nets.GaussianStochasticEncoder(X_DIM, Z_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())
        xyz_reduced_dim = self.x_encoder.get_output_dim() + self.y_encoder.get_output_dim() + self.Z_DIM
        self.v_given_xyz = nets.GaussianStochasticEncoder(xyz_reduced_dim, V_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU())

        # generative networks
        self.y_given_z = nets.StochasticDecoder(Z_DIM, Y_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)
        zv_reduced_dim = self.Z_DIM + self.V_DIM

        self.x_given_zv = nets.StochasticDecoder(zv_reduced_dim, X_DIM, hidden_dims=[200, 200], hidden_acts=nn.ReLU(), output_act=None)


        
    def forward(self,
                x, 
                y,
                n_samples_z=1,
                n_samples_v=1,
                inner_method='iwae',
                outer_method='elbo',
                ):
        """
        first estimator: $VAE\left(q(z|x), IWAE(q(v|x,y,z),p(x,y,z,v))\right)$


        For repeating random samples, I repeat the input data. I don't know if this is faster
        However, I need to repeat input data to compute likelihood regardless
        """
        
        # for k particles of z, repeat x k times
        x = repeat_newdim(x, n_samples_z, -2)

        z_outer, mu_z_given_x, var_z_given_x, log_posterior_z_given_x = self.z_given_x(x)

        # for n_samples_v particles of v, repeat z n_samples_v times, x_re k times and y n_samples_z * n_samples_v times
        #TODO: very akward, better way?
        x = repeat_newdim(x, n_samples_v, -2)
        z_inner = repeat_newdim(z_outer, n_samples_v, -2)
        y = repeat_newdim(y, n_samples_z, -2)
        y = repeat_newdim(y, n_samples_v, -2)

        # thus, for each (x,y) pair, there are {n_samples_z * n_samples_v} v particles
        # there are z_outer and z_inner for the fact that log_q(z_outer|x) in the outer sum can be used right away
        x_reduced = self.x_encoder(x)
        y_reduced = self.y_encoder(y)
        xyz_reduced = torch.cat((x_reduced, y_reduced, z_inner), dim=-1)
        v, mu, var, log_posterior_v_given_xyz = self.v_given_xyz(xyz_reduced)
        log_prior_z = log_normal_prior(z_inner)
        log_prior_v = log_normal_prior(v)

        zv_reduced = torch.cat((z_inner, v), dim=-1)
        x_recon = self.x_given_zv(zv_reduced)
        y_recon = self.y_given_z(z_inner)

        img_log_likelihood_x = log_prob_bernoulli(x_recon, x)
        img_log_likelihood_y = log_prob_bernoulli(y_recon, y)
        
        # for iwae, average over particles
        # for elbo, sum over weighted particles

        # compute iwae q(v|x,y,z) | p(x,y,z,v)
        # p(x,y,z,v) = p(y|z)p(x|z,v)p(z)p(v)

        img_log_likelihood_x.squeeze_(dim=-1)
        img_log_likelihood_y.squeeze_(dim=-1)
        log_prior_z.squeeze_(dim=-1)
        log_prior_v.squeeze_(dim=-1)
        log_posterior_v_given_xyz.squeeze_(dim=-1)
        log_posterior_z_given_x.squeeze_(dim=-1)

        inner_lowerbound = img_log_likelihood_x + img_log_likelihood_y + log_prior_z + log_prior_v - log_posterior_v_given_xyz
        if inner_method == 'iwae':
            inner_lowerbound = get_importance_bound(inner_lowerbound)
            inner_lowerbound = inner_lowerbound.sum(dim=-1, keepdim=True)
        elif inner_method == 'elbo':
            inner_lowerbound = inner_lowerbound.mean(dim=-1, keepdim=True)
        inner_lowerbound.squeeze_(-1)

        outer_lowerbound = inner_lowerbound - log_posterior_z_given_x
        if outer_method== 'iwae':
            outer_lowerbound = get_importance_bound(outer_lowerbound)
            outer_lowerbound = outer_lowerbound.sum(dim=-1, keepdim=True)
        elif outer_method == 'elbo':
            outer_lowerbound = outer_lowerbound.mean(dim=-1, keepdim=True)
        outer_lowerbound.squeeze_(-1)


        return outer_lowerbound
if __name__ == '__main__':
    pass
    a = torch.Tensor([[1.0, 2.0, 3.0], 
                    [4.0, 5.0, 6.0]])
    b = torch.LongTensor([1, 2])
    b = b.view(-1, 1)
    print(a.shape)
    print(b.shape)

    a = a.unsqueeze(1).repeat(1, 2, 1)
    b = b.unsqueeze(1).repeat(1, 2, 1)
    print(a.shape)
    print(b.shape)
    c = torch.exp(log_prob_softmax(a, b))
    # print(torch.exp(log_prob_softmax(a, b)))
    exit(0)