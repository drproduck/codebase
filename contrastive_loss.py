import os, sys
## Setting up paths
os.sys.path.append('/vinai/khiempd1/gwae')
import torch
from util.distance import batch_eudist_sq
from codebase.torchutils import expand_newdim
from codebase.distance import batch_eudist_sq

def contrastive_loss_fn(n_source, tau):
    """
    """
    def loss(z, y):
        """
        z: [batch_source, z_dim]
        y: label
        """
        dv = z.device
        dtype = z.dtype
        batch_source = z.shape[0]
        z_norm = norm2(z)
        z_unit = z / z_norm
        D = z_unit @ z_unit.t() / tau # [batch_source, batch_source]

        cost = 0.
        for i in range(n_source):
            len_i = (y == i).sum()
            interclass = D[y == i, :][:, y != i] # shape = [len_i, batch_source - len(i)]
            intraclass = D[y == i, :][:, y == i] # shape [len_i , len_i]
            interclass_xp = expand_newdim(interclass, len_i, 1) # [len_i, len_i, batch_source - len(i)]
            intraclass_xp = intraclass.unsqueeze(-1) # [len_i, len_i, 1]
            
            denom = torch.cat((interclass_xp, intraclass_xp), dim=-1) # [len_i, len_i, batch_source - len(i) + 1]
            log_denom = torch.logsumexp(denom, dim=-1, keepdim=False) # [len_i, len_i]
            class_cost = -1. * (intraclass - log_denom).sum()
            cost += class_cost / len_i ** 2
        return cost / n_source
    
    return loss

def contrastive_loss_C(C, N, k):
    """
    given matrix C of positive/negative (k negatives each row), derive contrastive loss
    C: cost matrix (logits)
    N: positive indices (bool)
    k: number of positive samples
    """
    n, m = C.shape[0], C.shape[1]
    neg_logits = C[~N].reshape(n, m - k) 
    neg_logits_xp = expand_newdim(neg_logits, k, 1) # [n, k, m - k]
    pos_logits = C[N].reshape(n, k)
    pos_logits_xp = pos_logits.unsqueeze(-1) # [n, k, 1]
    denom = torch.cat((pos_logits_xp, neg_logits_xp), dim=-1) # [n, k, m - k + 1]
    log_denom = torch.logsumexp(denom, dim=-1, keepdim=False)  # [n, k]
    cost = -1. * (pos_logits - log_denom).mean()
    
    return cost

def contrastive_loss_N(samples, pos):
    """
    given samples and (k) positives for each sample, derive contrastive loss
    the other samples are used as negatives
    samples: [batch x dim]
    pos: [batch x k x dim]
    k: number of positives per sample
    """
    k = pos.shape[1]
    n = samples.shape[0]
    neg_logits = -1. * batch_eudist_sq(samples, samples) # [n n]
    neg_logits[torch.arange(n), torch.arange(n)] = 0
    
    samples_norm = torch.sum(samples**2, dim=-1, keepdim=True) # [n, 1]
    pos_norm = torch.sum(pos**2, dim=-1, keepdim=False) # [n, k]
    samples_pos_dot = torch.sum(samples.unsqueeze(-2) * pos, dim=-1, keepdim=False) # [n, k]
    pos_logits = -1. * (samples_norm - 2 * samples_pos_dot + pos_norm) # [n k]
    neg_logits_xp = expand_newdim(neg_logits, k, 1) # [n, k, n]
    pos_logits_xp = pos_logits.unsqueeze(-1) # [n, k, 1]
    denom = torch.cat((pos_logits_xp, neg_logits_xp), dim=-1) # [n, k, n + 1]
    log_denom = torch.logsumexp(denom, dim=-1, keepdim=False) # [n, k]
    cost = -1. * (pos_logits - log_denom).mean()
    
    return cost

if __name__ == '__main__':
    C = torch.rand(3,3)
    N = torch.Tensor([[1,1,0],[0,1,1],[1,0,1]]).type(torch.bool)
    print(contrastive_loss(C, N, 2))
    
    
