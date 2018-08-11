"""
ops and Modules useful for exploring DeepSets, PointNets, DeepSetSSL
and all these fun variants
"""

import torch
from torch import nn
from torch.distributions import Categorical

sigmoid = torch.sigmoid
softmax = nn.functional.softmax


def tmax(x, dim):
    """return the max along dim dim in x without also returning argmax"""
    return x.max(dim=dim)[0]


def tmean(x, dim):
    return x.mean(dim)


def tsum(x, dim):
    return x.sum(dim)


def noisy_or(inputs, dim=0, squash=sigmoid):
    """noisy-or operator"""
    x = squash(inputs)
    return 1. - (1. - x).prod(dim)


def _isr_v(inputs, squash=sigmoid):
    """sub-function of isr (integrated segmentation and recognition) op"""
    x = squash(inputs)
    return x / (1. - x)


def isr(inputs, dim=0, squash=sigmoid):
    """integrated segmentation and recognition) op"""
    sum_v = _isr_v(inputs, squash).sum(dim)
    return sum_v / (1. + sum_v)


def log_sum_exp(inputs, dim=0, squash=sigmoid, r=1.):
    """log-sum-exp op"""
    x = squash(inputs)
    expd = x.mul(r).exp()
    return expd.mean(dim).log().mul(1. / r)


def tile(a, dim, n_tile):
    """tweaked https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4"""
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
    """make a 2d coordinate grid, with option to scale coords"""
    ch = torch.arange(height)
    if hscale:
        ch = scale(ch, *hscale)
    cw = torch.arange(width)
    if wscale:
        cw = scale(cw, *wscale)
    rep_height = tile(ch, 0, width)
    rep_width = cw.repeat(height)
    return torch.stack([rep_height, rep_width], 1)


def image_to_set(image, return_locations=False, hscale=None, wscale=None):
    """
    go from CHW image to set representation of image.
    if return_coords, will also return locations for each pixel in set
    scaled by hscale and wscale
    """
    if not return_locations:
        assert hscale is None,\
            'return_locations must be True if passing hscale'
        assert wscale is None,\
            'return_locations must be True if passing wscale'

    num_channels, height, width = image.shape

    observations = image.view(num_channels, -1)\
                        .t()\
                        .contiguous()

    if return_locations:
        locations = coord_grid(height, width, hscale, wscale)
        return (observations, locations)
    else:
        return observations


class SetMixedPooling(nn.Module):
    def __init__(self,
                 pooling_functions,
                 initial_weights=None,
                 dim_input=None,
                 learnable_weights=True,
                 weights_transform=None):
        """
        Takes inputs in expected set-representation format, runs pooling
        over the observation dimension with functions in pooling_functions,
        and weights the pooled activations for each dim of dim_input
        by weights.  The hope is to learn the most useful pooling function
        per feature.

        expects input of size batch_size x n_obs_per_input x dim_input
        """
        super().__init__()

        if dim_input is None:
            assert initial_weights is not None
            dim_input = initial_weights.shape[1]
        if initial_weights is None:
            initial_weights = torch.randn(len(pooling_functions), dim_input)

        assert len(initial_weights.shape) == 2
        assert len(pooling_functions) == initial_weights.shape[0]
        assert dim_input == initial_weights.shape[1]

        self.dim_input = dim_input
        self.num_pooling_functions = len(pooling_functions)
        self.pooling_functions = pooling_functions
        self.learnable_weights = learnable_weights
        self.weights = nn.Parameter(torch.tensor(initial_weights,
                                    requires_grad=self.learnable_weights))
        self.dim_input = initial_weights.shape[1]

        self.weights_transform = weights_transform

    def forward(self, inputs):
        assert inputs.dim() == 3,\
            'expects input of rank 3, with batch_size before dim_input'
        weights = self.weights
        if self.weights_transform:
            weights = self.weights_transform(weights)

        pooled = [pool(inputs, 1)
                  for pool in self.pooling_functions]

        # shape batch_size x num_pooling_functions x dim_input
        pooled = torch.stack(pooled, -2)
        return pooled.mul(self.weights).sum(-2)


class SoftmaxSetMixedPooling(SetMixedPooling):
    # TODO: Doc String
    def __init__(self,
                 pooling_functions,
                 initial_weights=None,
                 dim_input=None,
                 learnable_weights=True):

        def _weight_softmax(weights):
            return softmax(weights, 1)

        super().__init__(pooling_functions=pooling_functions,
                         initial_weights=initial_weights,
                         dim_input=dim_input,
                         learnable_weights=learnable_weights,
                         weights_transform=_weight_softmax)

        self.weights_distrs = Categorical(logits=self.weights)

    def weight_entropy(self):
        return self.weights_distrs.entropy()


class DeepSetSSL(nn.Module):
    def __init__(self,
                 obs_encoder,
                 obs_loc_encoder,
                 pooling_function,
                 classifier,
                 loc_encoder=None,
                 locs=None):
        # TODO: Doc String
        super().__init__()
        assert (loc_encoder is None) != (locs is None)
        self.obs_encoder = obs_encoder
        self.loc_encoder = loc_encoder
        self.locs = locs
        self.n_obs = locs.shape[0] if locs is not None else None
        self.obs_loc_encoder = obs_loc_encoder
        self.pooling_function = pooling_function
        self.classifier = classifier

    def forward(self, obs, locs=None):
        # bs x n_obs x dim_obs
        bs, n_obs, dim_obs = obs.shape
        emb_obs = self.obs_encoder(obs.view(-1, dim_obs))
        if self.loc_encoder is not None:
            assert locs is not None
            emb_loc = self.loc_encoder(locs).repeat(bs, 1)
        else:
            assert n_obs == self.n_obs
            emb_loc = self.locs.repeat(bs, 1)

        emb_obs_loc = self.obs_loc_encoder(torch.cat([emb_obs, emb_loc], -1))\
                          .view(bs, n_obs, -1)
        hidden = self.pooling_function(emb_obs_loc)
        return self.classifier(hidden)


if __name__ == '__main__':
    from torchvision.transforms import ToTensor, Compose
    from pyt.utils.quick_experiment import quick_experiment

    # Build Model
    num_classes = 10

    dim_obs = 1  # greyscale channel
    dim_loc = 2  # x, y

    dim_emb_obs = 32
    dim_emb_loc = dim_loc
    dim_emb_obs_loc = 256

    obs_encoder = nn.Sequential(
        nn.Linear(dim_obs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, dim_emb_obs))

    # loc_encoder = nn.Sequential()
    locs = coord_grid(28, 28, [-1., 1.], [-1., 1.])

    obs_loc_encoder = nn.Sequential(
        nn.Linear(dim_emb_obs + dim_emb_loc, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, dim_emb_obs_loc))

    classifier = nn.Sequential(
        nn.Linear(dim_emb_obs_loc, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, num_classes))

    pool_fns = [tmax, tmean, isr, log_sum_exp, noisy_or]
    pooling_function = SoftmaxSetMixedPooling(pooling_functions=pool_fns,
                                              dim_input=dim_emb_obs_loc)

    model = DeepSetSSL(obs_encoder=obs_encoder,
                       # loc_encoder=loc_encoder,
                       locs=locs,
                       obs_loc_encoder=obs_loc_encoder,
                       pooling_function=pooling_function,
                       classifier=classifier)

    set_transform = Compose([ToTensor(), image_to_set])

    quick_experiment(model=model,
                     dataset='mnist',
                     data_dir='/Users/jsb/datasets/',
                     task='classify',
                     transform=set_transform,
                     training_kwargs={'lr': 1e-1,
                                      'batch_size': 32,
                                      'weight_decay': 1e-5,
                                      'n_epoch': 6})
