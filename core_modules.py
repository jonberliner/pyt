import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from pyt.dropout import GaussianDropout
from pyt.normalization import GroupNorm

import os

cuda = torch.cuda.is_available()


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
        super().__init__()

        if trainable:
            assert self.dim_input is not None
            self.beta = F.softplus(Variable(torch.ones(self.dim_input)))
        else:
            self.beta = torch.FloatTensor([1.])

        if cuda:
            self.beta = self.beta.cuda()

    def forward(self, xs):
        return xs * F.sigmoid(self.beta.astype(xs) * xs)



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


# %%
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
        # self.dim_input = dim_input
        # self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.act_fn = act_fn
        self.norm_fn = norm_fns[str(norm)]
        self.p_drop = p_drop
        self.dropout_kind = dropout_kind
        self.drop_fn = drop_fns[str(dropout_kind)]

        _module_list = list()
        dim_in = dim_input
        for hi, dim_out in enumerate(dim_hidden):
            linear = nn.Linear(dim_in, dim_out)
            normed = self.norm_fn(dim_out)
            dropped = self.drop_fn(self.p_drop)

            _module_list += [linear, normed, dropped]
            dim_in = dim_out
        nlayer = len(dim_hidden) + 1
        linear = nn.Linear(dim_in, dim_output)
        _module_list.append(linear)
        self.module_list = nn.ModuleList(_module_list)

    def forward(self, xs):
        output = xs
        for layer in self.module_list:
            output = layer(output)
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    from torch.optim import Adam


    DATA_DIR = '/Users/jsb/data/mnist/'
    os.makedirs(DATA_DIR, exist_ok=True)

    toTensor = transforms.ToTensor()

    def transform_mnist(image):
        out = toTensor(image)
        out = out.view(784).float()
        return out

    dataset = MNIST(DATA_DIR, 
                    download=False, 
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

    # # %%
    # xs = Variable(data.train_data[:10]).float().mul(1./255.).view(-1, 784)
    model = SNMLP(784, 10, [128]*5)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # ys = model(xs)

    cel = nn.CrossEntropyLoss()

    def step(xs, ys, i_step=None):
        xs = Variable(xs, requires_grad=False)
        ys = Variable(ys, requires_grad=False)

        model.train()
        opt.zero_grad()
        yhl = model(xs)
        loss = cel(yhl, ys)

        loss.backward()
        opt.step()
        loss = loss.data.numpy()[0]
        if i_step is not None:
            print(f'(step {i_step}) loss: {loss}', end='\r')

    def evaluate(data_loader, split, i_step):
        nsam = 0
        model.eval()
        losses, accs = [], []
        for i_batch, xy in enumerate(data_loader):
            xs, ys = xy
            nsam += xs.shape[0]

            xs = Variable(xs, requires_grad=False)
            ys = Variable(ys, requires_grad=False)

            yhl = model(xs)
            loss = cel(yhl, ys).data.numpy()[0]

            _, yh = yhl.max(1)
            acc = np.mean(yh.data.numpy() == ys)

            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)
        print(f'({split} step {i_step}) acc: {acc}, loss: {loss}')

    i_step = 0
    for i_epoch in range(N_EPOCH):
        for xs, ys in data_loader:
            step(xs, ys, i_step=i_step)
            i_step += 1

        # indices = np.random.choice(50000, 1000, replace=False)
        # xs = dataset.train_data.in


