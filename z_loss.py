import torch
from torch import nn
from torch.distributions import Beta
from torch.nn.functional import softplus
from numpy import nan


lgamma = torch.lgamma


def incomplete_beta_function(x, a, b, max_iter=200):
    return torch.stack([_incomplete_beta_function(elt, a, b, max_iter)
                        for elt in x.view(-1)])\
                .view(x.shape)


def _incomplete_beta_function(x, a, b, max_iter=200):
    TINY = 1e-12
    STOP = 1e-8

    if x > (a + 1.) / (a + b + 2.):
        return 1. - _incomplete_beta_function(b, a, 1. - x)
    lbeta_ab = lgamma(a) + lgamma(b) + lgamma(a + b)
    front = (x.log().mul(a)
             + (1. - x).log().mul(b)
             - lbeta_ab)\
            .exp()\
            .div(a)
    f, c, d = 1., 1., 0.
    for i in range(max_iter):
        m = i * 0.5
        if i == 0:
            numer = 1.
        elif i % 2 == 0:
            numer = (m*(b-m)*x) / ((a+2.*m-1.) * a+2.*m)
        else:
            numer = -((a+m) * (a+b+m) * x) / ((a+2.*m) * (a+2.*m+1.))
        d = 1. + numer * d
        d = torch.tensor(max(d, TINY)).pow(-1)
        c = 1. + numer / c
        c = torch.tensor(max(c, TINY))
        cd = c * d
        f *= cd

        if abs(1. - cd) < STOP:
            return front * (f - 1.)
    return torch.tensor(nan)


# def incomplete_beta_function(a, b, inputs):
#     """
#     https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
#     incompbeta(a,b,x) evaluates incomplete beta function, here a, b > 0 and
#     0 <= x <= 1. This function requires contfractbeta(a,b,x, ITMAX = 200)
#     (Code translated from: Numerical Recipes in C.)
#     """

#     term1 = lgamma(a + b) - lgamma(a) - lgamma(b)
#     outputs = list()
#     for elt in inputs.view(-1):
#         if elt == 0. or elt == 1.:
#             output = elt
#         else:
#             lelt = elt.log()
#             lbeta = term1 + lelt.mul(a) + lelt.mul(b)
#             if elt < (a + 1.) / (a + b + 2.):
#                 output = lbeta.exp()\
#                               .mul(contfractbeta(a, b, elt))\
#                               .div(a)
#             else:
#                 output = 1. - lbeta.exp()\
#                                    .mul(contfractbeta(b, a, 1. - elt))\
#                                    .div(b)
#         outputs.append(output)
#     return torch.stack(outputs).view(inputs.shape)


# def contfractbeta(a, b, x, ITMAX=200):
#     """ contfractbeta() evaluates the continued fraction form of the incomplete
#     Beta function; incompbeta().
#     (Code translated from: Numerical Recipes in C.)"""

#     EPS = 1e-5
#     bm = az = am = 1.0
#     qab = a+b
#     qap = a+1.0
#     qam = a-1.0
#     bz = 1.0-qab*x/qap

#     for i in range(ITMAX+1):
#         em = float(i+1)
#         tem = em + em
#         d = em*(b-em)*x/((qam+tem)*(a+tem))
#         ap = az + d*am
#         bp = bz+d*bm
#         d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
#         app = ap+d*az
#         bpp = bp+d*bz
#         aold = az
#         am = ap/bpp
#         bm = bp/bpp
#         az = app/bpp
#         bz = 1.0
#         if torch.abs(az - aold) < (EPS * torch.abs(az)):
#             return az
#     _msg = ("could not get good approximation of incomplete beta function "
#             "in continued fraction form")
#     msg = f'{_msg} for a={a}, b={b}, x={x} after {ITMAX} iterations'
#     raise ValueError(msg)


def monotonic_fn(z, a, b):
    numer = lgamma(z + 1) + lgamma(z - a - b + 1)
    denom = lgamma(z - a + 1) + lgamma(z - b + 1)
    return numer - denom


class BetaLoss(nn.Module):
    def __init__(self,
                 size_average=True,
                 reduce=True):
        self.size_average = size_average
        self.reduce = reduce
        self.logit_alpha = nn.Parameter(torch.tensor(2.))
        self.logit_beta = nn.Parameter(torch.tensor(2.))
        self.beta_distr = Beta(softplus(self.logit_alpha),
                               softplus(self.beta))

    def _normalize(self, inputs):
        row_max = inputs.max(dim=1)[0]
        row_min = inputs.min(dim=1)[0]
        row_range = row_max - row_min
        ninputs = inputs.sub(row_min).div(row_range)
        return ninputs.div(ninputs.sum(dim=1))

    def forward(self, inputs, targets):
        self._check_valid_args(inputs, targets)
        ninputs = self._normalize(inputs)

        target_scores = inputs.gather(1, targets.unsqueeze(1))

        z_target_scores = (target_scores - mu) * sd.pow(-1.)
        z_losses = self.scale.pow(-1.)\
                   * softplus(self.scale * (self.loc - z_target_scores))
        if self.reduce:
            if self.size_average:
                output = z_losses.mean()
            else:
                output = z_losses.sum()
        else:
            output = z_losses
        return output

    def _check_valid_args(self, inputs, targets):
        assert inputs.dim() == 2
        assert targets.dim() == 1

        assert inputs.shape[0] == targets.shape[0]

        assert isinstance(inputs, torch.FloatTensor)
        assert isinstance(targets, torch.LongTensor)


class ZLoss(nn.Module):
    def __init__(self,
                 size_average=True,
                 reduce=True,
                 loc=None,
                 scale=None):
        self.size_average = size_average
        self.reduce = reduce
        loc = loc or nn.tensor(0.)
        scale = scale or nn.tensor(1.)
        self.loc = nn.Parameter(loc)
        self.scale = nn.Parameter(scale)

    def forward(self, inputs, targets):
        self._check_valid_args(inputs, targets)

        mu = inputs.mean(dim=1)
        sd = inputs.std(dim=1)

        target_scores = inputs.gather(1, targets.unsqueeze(1))
        z_target_scores = (target_scores - mu) * sd.pow(-1.)
        z_losses = self.scale.pow(-1.)\
                   * softplus(self.scale * (self.loc - z_target_scores))
        if self.reduce:
            if self.size_average:
                output = z_losses.mean()
            else:
                output = z_losses.sum()
        else:
            output = z_losses
        return output

    def _check_valid_args(self, inputs, targets):
        assert inputs.dim() == 2
        assert targets.dim() == 1

        assert inputs.shape[0] == targets.shape[0]

        assert isinstance(inputs, torch.FloatTensor)
        assert isinstance(targets, torch.LongTensor)
