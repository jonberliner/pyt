import torch
from torch import nn
import torch.functional as F


def noisy_or(inputs, dim=0, squash=torch.sigmoid):
    """noisy-or operator"""
    x = squash(inputs)
    return 1. - (1. - x).prod(dim)


def _isr_v(inputs, squash=torch.sigmoid):
    """sub-function of isr (integrated segmentation and recognition) op"""
    x = squash(inputs)
    return x / (1. - x)


def isr(inputs, dim=0, squash=torch.sigmoid):
    """integrated segmentation and recognition) op"""
    sum_v = _isr_v(inputs, squash).sum(dim)
    return sum_v / (1. + sum_v)


def log_sum_exp(inputs, dim=0, squash=torch.sigmoid, r=1.):
    """log-sum-exp op"""
    x = squash(inputs)
    expd = x.mul(r).exp()
    return expd.mean(dim).log().mul(1. / r)


def mixed_pooling(inputs,
                  weights,
                  pooling_functions,
                  pool_dim,
                  stack_dim,
                  mixing_dim):
    """
    apply pooling functions in pooling_functions along pool_dim,
    then aggregate weighted for each pooling function along mixing_dim
    Args:
        inputs (nd tensor): inputs
        weights ((n-1)d tensor): should be size of pool(inputs, dim)
        pooling_functions ([fn]): functions that reduce inputs
            on dim dim with function call form pool(inputs, dim=dim)
        pool_dim (int): dim to reduce inputs on
        stack_dim (int): dim to stack pooled inputs along
        mixing_dim (int): dim to mix pooled outputs over
    """
    # assert weights.shape[0] == len(pool_fns)
    # assert len(inputs) == 3, 'currently only supports 3d input'
    # assert len(weights) == 2, 'currently only supports 2d weights'
    pooled = torch.stack([pool(inputs, dim) for pool in pooling_functions],
                         dim=-1)
    return pooled.mul(weights.t()).sum(mixing_dim)


class MixedPooling(nn.Module):
    def __init__(self,
                 pooling_functions,
                 dim,
                 initial_weights,
                 learnable_weights=True,
                 weights_transform=None):
        super().__init__()
        self.pooling_functions = pooling_functions
        self.dim = dim
        # assert initial_weights.shape[0] == len(self.pooling_functions)
        # assert len(initial_weights.shape) == 2,\
        #     'currently only supports n_pooling_functions x dim_input weights'
        self.dim_input = initial_weights.shape[1]

        self.learnable_weights = learnable_weights
        self.weights = torch.tensor(initial_weights,
                                    requires_grad=self.learnable_weights)
        self.weights_transform = weights_transform

    def forward(self, inputs):
        weights = self.weights
        if self.weights_transform:
            weights = self.weights_transform(self.weights)
        return mixed_pooling(inputs=inputs,
                             dim=self.dim,
                             pooling_function=self.pooling_functions,
                             weights=weights)


def tile(a, dim, n_tile):
    """tweaked from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4"""
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cat([init_dim * torch.arange(n_tile).long() + i
                             for i in range(init_dim)])
    return torch.index_select(a, dim, order_index)


def scale(x, mn, mx):
    """scale x between mn and mx"""
    xmn = x.min()
    xmx = x.max()
    x = (x + xmn) / (xmx - xmn)
    x = x * (mx - mn) + mn
    return x


def coord_grid(height, width, hscale=None, wscale=None):
    ch = torch.arange(height)
    if hscale:
        ch = scale(ch, *hscale)
    cw = torch.arange(width)
    if wscale:
        cw = scale(cw, *wscale)
    rep_height = tile(ch, 0, width)
    rep_width = cw.repeat(height)
    return torch.stack([rep_height, rep_width], 1)


def images_to_sets(images, hscale=[-1., 1.], wscale=[-1., 1.]):
    """go from NCHW images of set representation of images"""
    batch_size, num_channels, height, width = images.shape
    locations = coord_grid(height, width, hscale, wscale)

    observations = images.view(batch_size, num_channels, -1)\
                         .tranpose(1, 2)\
                         .contiguous()
    return (locations, observations)


class SetMixedPooling(nn.Module):
    def __init__(self,
                 pooling_functions,
                 initial_weights=None,
                 dim_input=None,
                 learnable_weights=True,
                 weights_transform=None):
        """expects input of size batch_size x n_obs_per_input x dim_input"""
        super().__init__()

        if dim_input is None:
            assert initial_weights is not None
            dim_input = initial_weights.shape[1]
        if initial_weights is None:
            initial_weights = torch.randn(len(pooling_functions), dim_input)

        assert len(initial_weights.shape) == 0
        assert len(pooling_functions) == initial_weights.shape[1]
        assert dim_input == initial_weights.shape[1]

        self.dim_input = dim_input
        self.num_pooling_functions = len(pooling_functions)
        self.pooling_functions = pooling_functions
        self.learnable_weights = learnable_weights
        self.weights = torch.tensor(initial_weights,
                                    requires_grad=self.learnable_weights)
        self.dim_input = initial_weights.shape[1]

        self.learnable_weights = learnable_weights
        self.weights = torch.tensor(initial_weights,
                                    requires_grad=self.learnable_weights)
        self.weights_transform = weights_transform

    def forward(self, inputs):
        assert inputs.shape == 3,\
            'expects input of rank 3, with batch_size before dim_input'
        weights = self.weights
        if self.weights_transform:
            weights = self.weights_transform(self.weights)

        pooled = [pool(inputs, 1)
                  for pool in self.pooling_functions]

        # shape batch_size x num_pooling_functions x dim_input
        pooled = torch.stack(pooled, -2)
        return pooled.mul(self.weights).sum(-2)

        return mixed_pooling(inputs=inputs,
                             dim=self.dim,
                             pooling_function=self.pooling_functions,
                             weights=weights)


class SoftmaxMixedPooling(MixedPooling):
    def _weight_softmax(weights):
        return nn.functional.softmax(weights, dim=0)

    def __init__(self,
                 pooling_functions,
                 dim,
                 initial_weights,
                 learnable_weights=True):
        super().__init__(pooling_functions=pooling_functions,
                         dim=dim,
                         weights=weights,
                         learnable_weights=learnable_weights,
                         weights_transform=self._weight_softmax)
