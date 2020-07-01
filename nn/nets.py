from torch import nn
import torch
import torch.nn.functional as F
from codebase.nn import blocks
import typing
from torch import Tensor
from codebase.mathutils import log_prob_normal


class Net():

    def get_output_dim(self):
        return NotImplementedError

class StochasticNet(Net):

    def log_pdf(self):
        return NotImplementedError

# class LinearBlock(Net):
#     def __init__(self, d1, d2, act=nn.ReLU()):
#         super().__init__()
#         self.d1 = d1
#         self.d2 = d2
#         self.linear = nn.Linear(d1, d2)
#         self.act = act
    
#     def forward(self, x):
#         x = self.main(x)
#         if act is not None:
#             return self.act(x)
#         else:
#             return x

#     def get_output_dim(self):
#         return self.d2

class LinearMap(nn.Module, Net):
    def __init__(self, x_dim, hidden_dims, z_dim, act=nn.ReLU()):
        """
        Multiple Linear modules stacked on top of each other with a common activation in-between and after the last layer
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        # 3 cases: len(hidden_dims) > 1, = 1, = 0. need to be careful!
        if len(hidden_dims) > 0:
            if len(hidden_dims) == 1:
                pair_dims = [(x_dim, hidden_dims[0])]
            else: # len(hidden_dims) > 1
                pair_dims = zip([x_dim] + hidden_dims[:-1], hidden_dims)
            modules = [blocks.linear_block(d1, d2, act=act) for d1, d2 in pair_dims]

            modules.append(blocks.linear_block(hidden_dims[-1], z_dim, act=act))
            self.hidden_layers = nn.Sequential(*modules)

        else: # hidden_dims = []
            self.hidden_layers = blocks.linear_block(x_dim, z_dim, act=act)

    def forward(self, x):
        x = self.hidden_layers(x)
        return x

    def get_output_dim(self):
        return self.z_dim
        
    

class GaussianStochasticEncoder(nn.Module, StochasticNet):
    def __init__(self, x_dim, z_dim, hidden_dims=None, encoder: Net=None, hidden_acts=nn.ReLU()):
        super().__init__()
        if not ((hidden_dims is None) ^ (encoder is None)):
            raise Exception('hidden_dims and encoder are mutually exclusive')

        self.x_dim = x_dim
        self.z_dim = z_dim
        if encoder is not None:
            self.encoder = encoder
        else:
            encoder_output_dim = hidden_dims[-1]
            encoder_hidden_dims = hidden_dims[:-1]
            self.encoder = LinearMap(x_dim, encoder_hidden_dims, encoder_output_dim, act=hidden_acts)

        second2last_dim = self.encoder.get_output_dim()
        self.dense_mu = nn.Linear(second2last_dim, z_dim)
        self.dense_var = nn.Linear(second2last_dim, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        mu = self.dense_mu(z)
        var = F.softplus(self.dense_var(z)) + 1e-5
        z = mu + torch.randn(mu.shape).cuda() * var.sqrt()
        log_q_z_given_x = log_prob_normal(z, mu, var)
        return z, mu, var, log_q_z_given_x

    def get_output_dim(self):
        return self.z_dim

# class ConcatGaussianStochasticEncoder(nn.Module, Net):
#     def __init__(self, z_dim, encoders: typing.Dict[str, LinearMap], hidden_dims, hidden_acts=nn.ReLU()):
#         super().__init__()
#         """
#         :arg z_dim: the last dimension or dimension of the embedding
#         :arg encoders: a dictionary of component encoders before concatenation
#         :arg hidden_dims: hidden layers from the concat layer to the last layer
#         """
#         # need to register encoders
#         self.encoders = {}
#         for input_name, encoder in encoders.items():
#             net_name = input_name

#             self.add_module(net_name, encoder)
#             self.encoders[net_name] = encoder


#         self.intermediate_x_dim = sum([encoder.get_output_dim() for encoder in self.encoders.values()])
#         concat_encoder = LinearMap(self.intermediate_x_dim, hidden_dims, z_dim, hidden_acts=hidden_acts, last_act=nn.ReLU())

#         self.concat_stochastic_encoder = GaussianStochasticEncoder(self.intermediate_x_dim, z_dim, encoder=concat_encoder)    

#     def forward(self, xs: typing.Dict[str, torch.Tensor]):
#         inter_xs = {}
#         for key, encoder in self.encoders.items():
#             inter_xs[key] = encoder(xs[key])
#         x = torch.cat(list(inter_xs.values()), dim=-1)
#         z, mu, var, log_q_z_given_x = self.concat_stochastic_encoder(x)
#         return z, mu, var, log_q_z_given_x

#     def get_output_dim(self):
#         return self.z_dim


class StochasticDecoder(nn.Module, Net):
    def __init__(self, z_dim, x_dim, hidden_dims=None, encoder: Net=None, hidden_acts=nn.ReLU(), output_act=None):
        """
        The code supports empty and None hidden_dims, but generally it should be avoided incase of bugs :)
        hidden_dims and encoder are mutually exclusive
        """
        super().__init__()
        # hidden_dims and encoder are mutually exclusive
        if not ((hidden_dims is None) ^ (encoder is None)):
            raise Exception('hidden_dims and encoder are mutually exclusive')

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        if encoder is not None:
            self.encoder = encoder
        else:
            if len(hidden_dims) >= 1:
                encoder_output_dim = hidden_dims[-1]
                encoder_hidden_dims = hidden_dims[:-1]
                self.encoder = LinearMap(z_dim, encoder_hidden_dims, encoder_output_dim, act=hidden_acts)

            elif len(hidden_dims) == 0:
                self.encoder = LinearMap(z_dim, [], x_dim, act=None)
                

        # If hidden_dims is not empty
        if encoder is not None or len(hidden_dims) > 0:
            second2last_dim = self.encoder.get_output_dim()

            self.dense_logits = nn.Linear(second2last_dim, x_dim)

            self.output_act = output_act

    def forward(self, z):
        if self.encoder is not None or (self.hidden_dims is not None and len(self.hidden_dims) > 0):
            x = self.encoder(z)
            x = self.dense_logits(x)

        else:
            x = self.encoder(z)

        if self.output_act is not None:
            return self.output_act(x)
        else: return x