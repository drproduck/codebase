import torch

def get_jacobian(model, batched_inp, out_dim):
    batch_size, in_dim = batched_inp.shape

    inp = batched_inp.unsqueeze(1)  # batch_size, 1, input_dim
    inp = inp.repeat(1, out_dim, 1)  # batch_size, output_dim, input_dim
    
    out = model(inp.reshape(-1, in_dim)).reshape(batch_size, out_dim, out_dim)
    grad_inp = torch.eye(out_dim).reshape(1, out_dim, out_dim).repeat(batch_size, 1, 1).cuda()

    jacobian = torch.autograd.grad(out, [inp], [grad_inp], create_graph=True, retain_graph=True)[0]

    return jacobian

def sample_sphere(batch_size, d):
    samples = torch.randn((batch_size, d))
    zero_idx = (samples == torch.zeros(d))
    if torch.sum(zero_idx) > 0:
        samples[zero_idx] = torch.randn(len(zero_idx), d)
    return samples / torch.sqrt(torch.sum(samples**2, dim=-1, keepdim=True))