from torch.autograd import Function
from torch.nn import Module


class RevGradFn(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input


revgrad = RevGradFn.apply

class RevGrad(Module):
    def __init__(self, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return revgrad(input_)
    

class NecroReLUFn(Function):
    """ 
    A function that outputs like a ReLU (x for x >= 0 or 0 otherwise) 
    but has gradient of an identity function (1)
    """
    @staticmethod
    def forward(ctx, input_):
        input_[input_ < 0] = 0
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
necro_relu = NecroReLUFn.apply

class NecroReLU(Module):
    def __init__(self, *args, **kwargs):
        """ 
        A function that outputs like a ReLU (x for x >= 0 or 0 otherwise) 
        but has gradient of an identity function (1)
        """
        super().__init__(*args, **kwargs)

    def forward(self, input_):
        return necro_relu(input_)