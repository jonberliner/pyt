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


def isr(inputs, dim=0, squash=sigmoid):
    """integrated segmentation and recognition) op"""
    x = squash(inputs)
    v = x / (1. - x)
    sum_v = v.sum(dim)
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
    ch = torch.arange(height).float()
    if hscale:
        ch = scale(ch, *hscale)
    cw = torch.arange(width).float()
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
                 locs=None,
                 learnable_locs=False,
                 p_subsample=1.,
                 subsample_same_locs=True):
        # TODO: Doc String
        super().__init__()
        assert (loc_encoder is None) != (locs is None)
        self.obs_encoder = obs_encoder
        self.loc_encoder = loc_encoder
        self.learnable_locs = learnable_locs
        self.locs = nn.Parameter(locs, requires_grad=self.learnable_locs)
        self.n_obs = locs.shape[0] if locs is not None else None
        self.obs_loc_encoder = obs_loc_encoder
        self.pooling_function = pooling_function
        self.classifier = classifier

        self.p_subsample = p_subsample
        self.subsample_same_locs = subsample_same_locs

    def _subsample(self, obs, locs, p_subsample,
                   subsample_same_locs=True):
        if p_subsample < 1.:
            n_all_obs = obs.shape[1]
            dim_obs = obs.shape[2]
            n_obs = max(int(n_all_obs * p_subsample), 1)

            if self.subsample_same_locs:
                # same location subsampling for all datapoints in batch
                inds = torch.randperm(n_all_obs)[:n_obs]
                obs = obs.index_select(1, inds.to(obs.device))
                locs = locs.index_select(0, inds.to(locs.device))
            else:
                raise ValueError('not ready yet!')
                # different subsampling per datapoint in batch
                inds = torch.stack([torch.randperm(n_all_obs)[:n_obs]
                                    for _ in range(obs.shape[0])])
                obs = obs.gather(1, inds.to(obs.device)\
                                        .unsqueeze(-1)\
                                        .expand(-1, -1, dim_obs))
                locs = locs.gather(0, inds.to(locs.device))
        return obs, locs


    def forward(self, obs, locs=None):
        locs = locs or self.locs
        assert locs is not None

        p_subsample = self.p_subsample if self.training else 1.
        obs, locs = self._subsample(obs, locs, p_subsample)

        # bs x n_obs x dim_obs
        bs, n_obs, dim_obs = obs.shape
        emb_obs = self.obs_encoder(obs.view(-1, dim_obs))

        if self.loc_encoder is not None:
            emb_loc = self.loc_encoder(locs).repeat(bs, 1)
        else:
            emb_loc = locs.repeat(bs, 1)

        emb_obs_loc = self.obs_loc_encoder(torch.cat([emb_obs, emb_loc], -1))\
                          .view(bs, n_obs, -1)

        hidden = self.pooling_function(emb_obs_loc)
        assert hidden.shape[0] == bs
        assert hidden.shape[1] == emb_obs_loc.shape[2]
        return self.classifier(hidden)


if __name__ == '__main__':
    import torch
    from torch import nn
    from torchvision.transforms import ToTensor, Compose, Normalize
    from pyt import sets
    from pyt.utils.quick_experiment import quick_experiment

    use_cuda = torch.cuda.is_available()
    # torch.device object used throughout this script# torch.devi
    device = torch.device("cuda" if use_cuda else "cpu")
    # Build Model
    num_classes = 10

    dim_obs = 1  # greyscale channel
    dim_emb_obs = dim_obs

    # observations will be their identity
    obs_encoder = nn.Sequential()

    # locs are concat of cartesian and polar coords
    locs = coord_grid(28, 28, [-1., 1.], [-1., 1.])
    rs = locs.pow(2).sum(1).sqrt().view(-1, 1)
    thetas = locs[:, 1].div(locs[:, 0]).atan().view(-1, 1)
    locs = torch.cat([locs, rs, thetas], 1)

    dim_loc = locs.shape[1]
    dim_emb_loc = dim_loc

    dim_emb_obs_loc = 512
    obs_loc_encoder = nn.Sequential(
        nn.Linear(dim_emb_obs + dim_emb_loc, 128),
        nn.ELU(),
        nn.Linear(128, 64),
        nn.ELU(),
        nn.Linear(64, dim_emb_obs_loc))

    # TODO: let's get the mixtures of perm-invar functions running well
    #     pool_fns = [sets.tmax,
    #                 sets.tmean,
    #                 sets.isr,
    #                 sets.log_sum_exp,
    #                 sets.noisy_or]
    #     pooling_function = sets.SoftmaxSetMixedPooling(
    #                                 pooling_functions=pool_fns,
    #                                 dim_input=dim_emb_obs_loc)
    # pooling_function = lambda x: sets.tmax(x, 1)
    pooling_function = lambda x: sets.noisy_or(x, 1)

    classifier = nn.Sequential(
        # nn.ELU(),
        nn.Linear(dim_emb_obs_loc, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, num_classes))

    model = sets.DeepSetSSL(obs_encoder=obs_encoder,
                            # loc_encoder=loc_encoder,
                            locs=locs,
                            learnable_locs=False,
                            obs_loc_encoder=obs_loc_encoder,
                            pooling_function=pooling_function,
                            classifier=classifier,
                            p_subsample=1.,
                            subsample_same_locs=True).to(device)

    set_transform = Compose([ToTensor(),
                             Normalize((0.1307,), (0.3081,)),
                             sets.image_to_set])

    model.p_subsample = 0.8
    quick_experiment(model=model,
                     dataset='mnist',
                     data_dir='/Users/jsb/datasets/',
                     p_splits={'train': 1.},
                     task='classify',
                     transform=set_transform,
                     epochs_per_eval=1,
                     print_training_loss_every=2,
                     training_kwargs={'lr': 1e-4,
                                      'batch_size': 32,
                                      'n_epoch': 100})
