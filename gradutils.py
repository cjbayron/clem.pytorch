# PyTorch Gradient Utility Functions

import numpy as np
import quadprog
import torch


def get_gradient(model):
    ''' Get current gradients of a PyTorch model.

    This collects ALL GRADIENTS of the model in a SINGLE VECTOR.
    '''
    grad_vec = []
    for param in model.parameters():
        if param.grad is not None:
            grad_vec.append(param.grad.view(-1))
        else:
            # Part of the network might has no grad, fill zero for those terms
            grad_vec.append(param.data.clone().fill_(0).view(-1))

    return torch.cat(grad_vec)


def update_gradient(model, new_grad):
    ''' Overwrite current gradient values in Pytorch model.

    This expects a SINGLE VECTOR containing all corresponding gradients for the model.
    This means that the number of elements of the vector must match the number of gradients in the model.
    '''
    ptr = 0
    for param in model.parameters():
        num_params = param.numel() # number of elements
        if param.grad is not None:
            # replace current param's gradients (in-place) with values from new gradient
            param.grad.copy_(new_grad[ptr:(ptr+num_params)].view_as(param))

        ptr += num_params


def project_gradient_qp(gradient, memories, margin=0.5, eps=1e-3):
    ''' Solves the GEM dual QP described in the paper given a proposed
    gradient "gradient", and a memory of task gradients "memories".
    Returns "gradient" with the final projected update.

    input:  gradient, p-vector
    input:  memories, (t * p)-vector
    output: proj, p-vector (projected gradient)

    Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
    Modified from: https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/agents/exp_replay.py#L119
    '''
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()

    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]

    proj = np.dot(v, memories_np) + gradient_np

    return torch.Tensor(proj).view(-1)
