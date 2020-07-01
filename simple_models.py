from torch import nn
import torch.nn.functional as F
from codebase.nn import blocks, nets


# define some simple modules for testing
class GaussianConcatStochasticEncoder(nn.Module):
    def __init__(self, z_dim, individual_hidden_dims: typing.Dict, 
                common_hidden_dims: typing.Dict, 
                input_dims: typing.Dict[str, List]
                ):

        self.modules= {}
        self.torch_modules = {}
        self.z_dim = z_dim
        for input_key, input_dim in input_dims:
            input_hidden_dim = individual_hidden_dims[input_key]
            pair_dims = [(d1, d2) for d1, d2 in zip([input_dim] + input_hidden_dim[:-1], input_hidden_dim)]
            input_module = [nn.block(d1, d2) for d1, d2 in pair_dims]
            self.modules[input_key] = input_module
            net_name = f'{input_key}_net'
            self.torch_modules[net_name] = nn.Sequential(*self.modules[input_key])
            setattr(self, net_name, self.torch_modules[net_name])
        
        # define concat net
        self.input_hidden_concat_dim = 0
        for hidden_dim in individual_hidden_dims.values():
            self.input_hidden_concat_dim += hidden_dim[-1]

        # pair_dims = [(d1, d2) for d1, d2 in zip([self.input_hidden_concat_dim] + common_hidden_dims, common_hidden_dims + [z_dim])]
        self.z_net = StochasticEncoder(common_hidden_dims, self.input_hidden_concat_dim, z_dim)


    def forward(self, inputs: typing.Dict):
        input_transforms = []
        for input_key, input_batch in inputs:        
            input_transform = self.torch_modules[input_key](input_batch)
            input_transforms.append(input_transform)
        hidden_concat = torch.cat(input_transforms, dim=-1)
        z, mu, var = self.z_net(hidden_concat)
        return z, mu, var
