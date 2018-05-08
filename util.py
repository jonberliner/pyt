import torch
from torch import nn
from torch.autograd import Variable
import os

CUDA = torch.cuda.is_available()

# # TODO: decide if want to move forward with decorators or keep nested functions
# def cuda_if_available(fn):
#     def wrapper(*args, **kwargs):
#         output = fn(*args, **kwargs)
#         if CUDA:
#             output = output.cuda()
#         return output
#     return wrapper

# @cuda_if_available
# def dt(array):
#     return torch.DoubleTensor(array)


def cuda_if_available(tensor):
    tensor = tensor.cuda() if CUDA else tensor
    return tensor


def dt(array):
    return cuda_if_available(torch.DoubleTensor(array))


def ft(array):
    return cuda_if_available(torch.FloatTensor(array))


def lt(array):
    return cuda_if_available(torch.LongTensor(array))


def bt(array):
    return cuda_if_available(torch.ByteTensor(array))


def vdt(array, requires_grad=False):
    return Variable(dt(array), requires_grad=requires_grad)


def vft(array, requires_grad=False):
    return Variable(ft(array), requires_grad=requires_grad)


def vlt(array, requires_grad=False):
    return Variable(lt(array), requires_grad=requires_grad)


def vbt(array, requires_grad=False):
    return Variable(bt(array), requires_grad=requires_grad)


def var_to_numpy(var):
    if var.is_cuda:
        var = var.cpu()
    return var.data.numpy()
