import torch
from torch import nn
from torch.autograd import Variable

# from https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.tensor(alpha)

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.training:
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            return x * epsilon
        else:
            return x


class VariationalDropout(nn.Module):
    def __init__(self, alpha):
        super(VariationalDropout, self).__init__()

        self.max_alpha = alpha
        # Initial alpha
        self.log_alpha = nn.Parameter(alpha.log())

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl =\
            0.5 * self.log_alpha\
            + c1 * alpha\
            + c2 * alpha**2\
            + c3 * alpha**3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.training:
            # N(0,1)
            epsilon = x.new_zeros(*x.shape).normal_(0., 1.)

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data,
                                              max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x


