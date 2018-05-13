import torch
from torch import nn
from torch.nn import functional as F

from pyt.dropout import GaussianDropout
from pyt.normalization import GroupNorm
from pyt.util import CUDA, var_to_numpy

import numpy as np

import os


class SizedModule(nn.Module):
    """wrapper around nn.Module with consistent input and output size attributes
    to make snapping networks together magically a little easier down-stream"""
    def __init__(self, dim_input=None, dim_output=None):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output


class ReadInReadOutModule(SizedModule):
    """module that takes care of read-in and read-out of data.  useful for
    easily swapping in hidden computations of interest, which is where most
    of the interesting modeling work occurs."""
    def __init__(self, dim_input, dim_output, hidden_encoder):
        """
        Args:
            dim_input (int): input size
            dim_output (int): output size
            hidden_encoder (nn.Modules): model that takes in linear readin.
                it's output is passed to a linear readout.
        """
        assert dim_input
        assert dim_output
        if hidden_encoder:
            assert hasattr(hidden_encoder, 'dim_input'), \
                ("if passing hidden_encoder, expects hidden_encoder to have"
                 "attribute dim_input, as in pyt's SizedModule")
            assert hasattr(hidden_encoder, 'dim_output'), \
                ("if passing hidden_encoder, expects hidden_encoder to have"
                 "attribute dim_output, as in pyt's SizedModule")
            assert hasattr(hidden_encoder, 'dim_input')
            assert hasattr(hidden_encoder, 'dim_output')
        super().__init__(dim_input, dim_output)

        self.hidden_encoder = hidden_encoder
        self.readin = nn.Linear(self.dim_input, self.hidden_encoder.dim_input)
        self.readout = nn.Linear(self.hidden_encoder.dim_output, self.dim_output)

    def forward(self, inputs):
        output = self.readin(inputs)
        output = self.hidden_encoder(output)
        output = self.readout(output)
        return output


class Swish(SizedModule):
    def __init__(self, dim_input=None, trainable=False):
        super().__init__(dim_input)
        self.trainable = trainable

        _dim_input = dim_input or 1
        _beta_logit = torch.ones(_dim_input) * 0.5412  # softplus ~= 1.
        if trainable:
            assert self.dim_input is not None
        self.beta_logit = nn.Parameter(_beta_logit,
                                       requires_grad=self.trainable)

    def forward(self, xs):
        beta = F.softplus(self.beta_logit)
        return xs * F.sigmoid(beta * xs)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, xs):
        return xs


norm_fns = {
    'None': Identity,
    'batch': nn.BatchNorm1d,
    'group': GroupNorm
}


drop_fns = {
    'None': Identity,
    'standard': nn.Dropout,
    'alpha': nn.AlphaDropout,
    'gaussian': GaussianDropout
}


class MLP(SizedModule):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden=[],
                 act_fn=F.leaky_relu,
                 p_drop=0.,
                 dropout_kind=None,
                 norm=None):
        super().__init__(dim_input, dim_output)
        self.dim_hidden = dim_hidden
        self.act_fn = act_fn
        if norm is None or isinstance(norm, str):
            self.norm_fn = norm_fns[str(norm)]
        else:
            self.norm_fn = norm
        self.p_drop = p_drop
        self.dropout_kind = dropout_kind
        self.drop_fn = drop_fns[str(dropout_kind)]

        # _module_list = list()
        _linears = list()
        _normeds = list()
        _droppeds = list()
        dim_in = dim_input
        for hi, dim_out in enumerate(dim_hidden):
            linear = nn.Linear(dim_in, dim_out)
            normed = self.norm_fn(dim_out)
            dropped = self.drop_fn(self.p_drop)

            _linears.append(linear)
            _normeds.append(normed)
            _droppeds.append(dropped)

            dim_in = dim_out
        self.readout = nn.Linear(dim_in, dim_output)

        self.linear = nn.ModuleList(_linears)
        self.normed = nn.ModuleList(_normeds)
        self.dropped = nn.ModuleList(_droppeds)

    def forward(self, xs):
        output = xs
        for lin, norm, drop in zip(self.linear, self.normed, self.dropped):
            output = drop(norm(self.act_fn(lin(output))))
        output = self.readout(output)
        return output


class SNMLP(MLP):
    def __init__(self, dim_input, dim_output, dim_hidden, p_drop=0.):
        super().__init__(dim_input=dim_input,
                         dim_output=dim_output,
                         dim_hidden=dim_hidden,
                         act_fn=F.selu,
                         norm=None,
                         dropout_kind='alpha',
                         p_drop=p_drop)


# class ConvEncoder(nn.Module):
#     def __init__(self,
#                  dim_input,
#                  dim_output,
#                  dim_hidden=[],
#                  act_fn=Swish(),
#                  norm_fn=Identity,
#                  separable=False):
#         super().__init__()
#         self.dim_input = dim_input
#         self.dim_output = dim_output
#         self.dim_hidden = dim_hidden

#         dim_in = self.dim_input
#         self.n_hid = len(self.dim_hidden)
#         self.net = nn.Sequential()
#         for hi, dim_out in enumerate(self.dim_hidden):
#             sep = separable and hi > 0
#             self.net.add_module(f'conv_{hi}', ConvLayer(dim_in, dim_out, (3, 3),
#                                                         padding=1, separable=sep,
#                                                         act_fn=act_fn, norm_fn=norm_fn))
#             self.net.add_module(f'downsample{hi}', nn.Conv2d(dim_out, dim_out, (3, 3), stride=2, padding=1))
#             dim_in = dim_out
#         dim_in = dim_out
#         self.net.add_module('readout', nn.Conv1d(dim_in, self.dim_output))

#         self.max_or_mean_logits = nn.Parameter(torch.randn(self.dim_output))

#     def forward(inputs):
#         vinputs = Variable(inputs)
#         logits = self.net(vinputs)
#         batch_size, ch_out = logits.shape[:2]
#         maxes = logits.view(batch_size, ch_out, -1).max(-1)
#         means = logits.view(batch_size, ch_out, -1).mean(-1)
#         w_maxormean = F.sigmoid(self.max_or_mean_logits).view(1, -1)
#         pooled = (w_maxormean * maxes) + ((1. - w_maxormean) * means)  # bs x chout
#         return pooled


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    from torch.optim import Adam


    DATA_DIR = '/home/j.s.berliner/mnist/'
    os.makedirs(DATA_DIR, exist_ok=True)

    toTensor = transforms.ToTensor()

    def transform_mnist(image):
        out = toTensor(image)
        out = out.view(784).float()
        return out

    dataset = MNIST(DATA_DIR,
                    download=True,
                    transform=transform_mnist,
                    train=True)
    data_loader = DataLoader(dataset,
                             batch_size=64,
                             shuffle=True)

    test_dataset = MNIST(DATA_DIR,
                         download=False,
                         transform=transform_mnist,
                         train=False)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=64,
                                  shuffle=False)

    model = SNMLP(784, 10, [128]*5)
    if torch.cuda.is_available():
        model = model.cuda()
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    cel = nn.CrossEntropyLoss()

    def step(xs, ys, i_step=None):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)
        if CUDA:
            xs, ys = xs.cuda(), ys.cuda()

        model.train()
        opt.zero_grad()
        yhl = model(xs)
        loss = cel(yhl, ys)

        loss.backward()
        opt.step()
        loss = var_to_numpy(loss)[0]
        if i_step is not None:
            print('(step {:d}) loss: {:.3f}'.format(i_step, loss), end='\r')

    def evaluate(data_loader, split, i_step):
        nsam = 0
        model.eval()
        losses, accs = [], []
        for i_batch, xy in enumerate(data_loader):
            xs, ys = xy
            nsam += xs.shape[0]

            xs = Variable(xs, requires_grad=False)
            ys = Variable(ys, requires_grad=False)
            if CUDA:
                xs, ys = xs.cuda(), ys.cuda()

            yhl = model(xs)
            loss = var_to_numpy(cel(yhl, ys))

            _, yh = yhl.max(1)
            acc = np.mean(var_to_numpy(yh == ys))

            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)
        print('({:s} step {:d}) acc: {:.3f}, loss: {:.3f}'\
               .format(split, i_step, acc, loss))

    i_step = 0
    for i_epoch in range(10):
        evaluate(data_loader, 'train', i_step)
        evaluate(test_data_loader, 'test', i_step)
        for xs, ys in data_loader:
            step(xs, ys, i_step=i_step)
            i_step += 1


