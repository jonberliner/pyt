import torch
from torch import nn
from torch.nn import functional as F


class SVDLinear(nn.Module):
    """
    linear layer (not affine) with params constituted by svd of the weight matrix.
    increases forward compute cost at the expense of not needing to compute svd
    """
    def __init__(self,
                 dim_input,
                 dim_output,
                 min_singular_value=1e-4,
                 U_seed=None,
                 V_seed=None):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_sigma = min(dim_input, dim_output)
        self._sigma_diag_logits = nn.Parameter(torch.randn(self.dim_sigma))

        U_seed = self.to_orthogonal_matrix_seed(U_seed, dim_output)
        V_seed = self.to_orthogonal_matrix_seed(V_seed, dim_input)

        self.U_seed = nn.Parameter(U_seed)
        self.V_seed = nn.Parameter(V_seed)
        self.rectifier = F.softplus
        self.min_singular_value = min_singular_value

    @staticmethod
    def to_orthogonal_matrix_seed(seed, dim):
        "ensure seed is an dimxdim torch tensor"
        if isinstance(seed, None):
            seed = torch.randn(dim, dim)
        if isinstance(seed, int):
            seed = torch.randn(seed, seed)
        elif isinstance(seed, np.array):
            seed = torch.tensor(seed)
        assert isinstance(seed, torch.Tensor)
        assert len(seed.shape) == 2
        assert seed.shape[0] == seed.shape[1]
        assert seed.shape[0] == dim
        return seed

    def to_orthogonal_matrix(self, seed):
        """
        return orthogonal matrix given arbitrary square matrix seed.
        random matrices are uniformly distributed according to the Haar measure
        as explained here: https://arxiv.org/pdf/math-ph/0609050.pdf
        """
        q, r = torch.qr(seed)
        d = r.diag()
        ph = d / d.abs()
        output = q @ ph.diag() @ q
        return output

    @property
    def sigma_diag(self):
        """
        return the diagonal of sigma.  ensures singular values are ordered
        monotonically decreasing, constrained positive, and clipped to 0
        if considered small
        """
        deltas = self.rectifier(self._sigma_diag_logits)
        diag = torch.flip(torch.cumsum(deltas, 0), (0,))  # reverse
        # clip small values to 0
        mask = diag < self.min_singular_value
        diag = diag.masked_fill(mask, 0.)
        return diag

    @property
    def Sigma(self):
        "sigma matrix of svd(W)"
        diag = self.sigma_diag
        sigma = torch.zeros(self.dim_output, self.dim_input)
        sigma[:self.dim_sigma, :self.dim_sigma] = diag.diag()
        return sigma

    @property
    def pinvSigma(self):
        "pseudoinverse of sigma"
        diag = self.sigma_diag
        # pseudo invert
        nonzero = diag.nonzero()[0]
        diag.scatter(0, nonzero, 1./diag[nonzero])
        pinv_sigma = torch.zeros(self.dim_output, self.dim_input)
        pinv_sigma[:self.dim_sigma, :self.dim_sigma] = diag.diag()
        return pinv_sigma

    @property
    def U(self):
        return self.to_orthogonal_matrix(self.U_seed)

    @property
    def V(self):
        return self.to_orthogonal_matrix(self.V_seed)

    @property
    def W(self):
        return self.U @ self.Sigma @ self.V.t()

    @property
    def pinvW(self):
        return self.V @ self.pinvSigma.t() @ self.U.t()

    @property
    def condW(self):
        "condition number of W.  The lower, the more robust to errors"
        return self.W.norm() * self.pinvW.norm()

    def forward(self, inputs):
        return inputs @ self.W.t()

    def invert(self, outputs):
        return outputs @ self.pinvW.t()


class Net(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden):
        super().__init__()
        self.fc1 = SVDWeight(dim_input, dim_hidden)
        self.fc2 = SVDWeight(dim_hidden, dim_output)
        self.act_fn = nn.SELU()

    def forward(self, inputs):
        return self.fc2(self.act_fn(self.fc1(inputs)))

    def condition_loss(self):
        return self.fc1.condW + self.fc2.condW

    def near01(self):
        out = 0.
        for W in [self.fc1.W, self.fc2.W]:
            out += ((W - 1.).abs().sum() + W.abs().sum()) / 2.
        return out


def near01(net):
    out = 0.
    for W in [net.fc1.W, net.fc2.W]:
        out += ((W - 1.).abs().sum() + W.abs().sum()) / 2.
    return out


from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

net = Net(784, 10, 128)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x.view(-1)])

train_dataset = MNIST(root='/Users/jsb/datasets/MNIST', transform=img_transform)
test_dataset = MNIST(root='/Users/jsb/datasets/MNIST', train=False, transform=img_transform)
data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

cel = nn.CrossEntropyLoss()

opt = torch.optim.Adam(params=net.parameters(), lr=1e-3)

for xs, ys in data_loader:
    net.train()
    opt.zero_grad()
    yhat = net(xs)
    loss = cel(yhat, ys)
    loss += net.condition_loss() * 1e-4
    loss += near01(net) * 1e-4
    loss.backward()
    opt.step()
    print(loss.item())

for epoch in range(10):
    net.train()
    for xs, ys in data_loader:
        opt.zero_grad()
        yhat = net(xs)
        loss = cel(yhat, ys)
        # loss += net.condition_loss() * 1e-4
        # loss += near01(net) * 1e-4
        acc = yhat.argmax(1).eq(ys).float().mean()
        cond_loss = net.condition_loss() * 1e-4
        zero_one_loss = near01(net) * 1e-4
        _loss = loss + cond_loss + zero_one_loss
        _loss.backward()
        opt.step()
        print(_loss.item())
        print(f"[epoch {epoch}]\n\tloss: {loss.item()}\n\tacc: {acc.item()}\n\tcond_loss:{cond_loss.item()}]n]t01loss: {zero_one_loss.item()}")

    net.eval()
    ii = 0
    for loader in [data_loader, test_data_loader]:
        ii += 1
        if ii == 1:
            print('TRAIN')
        else:
            print('TEST')
        jj = 0
        for xs, ys in loader:
            jj += 1
            if jj > 60:
                break
            with torch.no_grad():
                yhat = net(xs)
                loss = cel(yhat, ys)
                acc = yhat.argmax(1).eq(ys).float().mean()
                cond_loss = net.condition_loss() * 1e-4
                zero_one_loss = near01(net) * 1e-4
                print(f"[epoch {epoch}]\n\tloss: {loss.item()}\n\tacc: {acc.item()}\n\tcond_loss:{cond_loss.item()}]n]t01loss: {zero_one_loss.item()}")



    # def invert(self, outputs):

    #     out = outputs @ self.U.t()  # bsxdim_outut x dim_output x dim_output
    #     out = 1. / self.Sigma @ out

    #     sigma_diag = self.rectifier(self._sigma_diag_logits) + self.TINY

