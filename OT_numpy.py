from utils import *
from time import time


def get_B(args):
    """
    The entropic regularized optimal transport plan
    """
    return np.exp(- args.C / args.eta + args.u + args.v.T)

def f_dual(args):
    """
    the dual of the entropic-regularized balanced OT
    """
    f = np.sum(args.B) + dotp(args.u, args.r) + dotp(args.v, args.c)
    return f

def f_primal(args):
    """
    the primal of the entropic-regularized balanced OT
    """
    unreg_val = dotp(args.C, args.B)

    entropy = get_entropy(args.B)

    return unreg_val - args.eta * entropy

def unreg_f(args):
    """
    the unregularized objective with solutions u, v
    """
    return dotp(args.C, args.B)

def norm1_constraint(args):
    a = args.B.sum(axis=1).reshape(-1, 1)
    b = args.B.sum(axis=0).reshape(-1, 1)
    return norm1(a - args.r) + norm1(b - args.c)


def sinkhorn(r, c, C, eta=0.1, max_iter=50):
    start = time()
    """
    log stable balanced OT
    :arg C: cost matrix
    :arg r: first marginal
    :arg c: second marginal
    """

    # collect some stats
#     f_dual_val_list = []
#     f_primal_val_list = []
#     unreg_f_val_list = []
#     constraint_norm_list = []

    args = DictObject(C=C, r=r, c=c, B=None, u=None, v=None, eta=eta, max_iter=max_iter)
    # initial solutions
    args.u = np.zeros(r.shape)
    args.v = np.zeros(c.shape)
    args.B = get_B(args)

    # compute before any updates
#     f_dual_val = f_dual(args)
#     f_dual_val_list.append(f_dual_val)

#     unreg_f_val = unreg_f(args)
#     unreg_f_val_list.append(unreg_f_val)

#     f_primal_val = f_primal(args)
#     f_primal_val_list.append(f_primal_val)

#     rc_diff = norm1_constraint(args)
#     constraint_norm_list.append(rc_diff)

#     stop_iter = max_iter
    for i in range(max_iter):
        # update
        args.B = get_B(args)
        if i % 2 == 0:
            a = args.B.sum(axis=1).reshape(-1, 1)
            args.u = args.u + np.log(args.r) - np.log(a)
        else:
            b = args.B.sum(axis=0).reshape(-1, 1)
            args.v = args.v + np.log(args.c) - np.log(b)

#         f_dual_val = f_dual(args)
#         f_dual_val_list.append(f_dual_val)

#         unreg_f_val = unreg_f(args)
#         unreg_f_val_list.append(unreg_f_val)

#         f_primal_val = f_primal(args)
#         f_primal_val_list.append(f_primal_val)

#         rc_diff = norm1_constraint(args)
#         constraint_norm_list.append(rc_diff)

    alpha = eta * (args.u - args.u.sum() / len(r))
    alpha = alpha.flatten()

#     info = DictObject(f_dual_val_list=f_dual_val_list,
#                       unreg_f_val_list=unreg_f_val_list,
#                       f_primal_val_list=f_primal_val_list,
#                       constraint_norm_list=constraint_norm_list,
#                       alpha=alpha,
#                       stop_iter=stop_iter
#                      )
    
    
    print('sinkhorn {} iters: {}'.format(max_iter, time() - start))
    return args.B, unreg_f(args), alpha, args