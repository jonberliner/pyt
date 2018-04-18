from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from numpy import mean
import os
from torchvision.transforms import ToTensor

from pyt.util import CUDA, var_to_numpy


TASK_LOSS_FNS = {'autoencode': nn.BCEWithLogitsLoss,
                 'classify': nn.CrossEntropyLoss,
                 'regress': nn.MSELoss}

TASKS = list(TASK_LOSS_FNS.keys())


def test_over_mnist(model, 
                    data_dir, 
                    task, 
                    training_kwargs={},
                    transform=ToTensor(),
                    data_seed=None,
                    training_penalty=None):
    """quickly train a model over mnist
    Args:
        model (nn.Module): pytorch model to optimize
        data_dir (str): path to data_dir to (possibly download and) find mnist data
        task (str): type of mnist task.  must be in {:s}
        flatten (bool): flatten data (bs x 784) or not? (bs x 28 x 28)
        channel_dim (bool): add channel_dim? (bs x {ch} x ...)
        training_kwargs (dict; optional): kwargs for training
        custom_pen (nn.Module; optional): takes in custom penalties.
            takes in model, inputs, and outputs (all optional))"""
    assert task in TASKS, f'task must be in {str(tasks)}'
    n_epoch = training_kwargs.get('n_epoch', 10)
    batch_size = training_kwargs.get('batch_size', 64)
    lr = training_kwargs.get('lr', 1e-3)
    weight_decay = training_kwargs.get('weight_decay', 0.)
    # TODO: for testing over small data
    # n_train = training_kwargs.get('n_train', -1)
    # n_test = training_kwargs.get('n_test', -1)

    P_SPLITS = {'train': 1.}
    BALANCED = True

    data_loaders = quick_mnist(data_dir, 
                               batch_size, 
                               transform=transform, 
                               balanced=BALANCED,
                               seed=data_seed, 
                               p_splits=P_SPLITS)
    train_data_loader, test_data_loader = data_loaders['train'], data_loaders['test']

    loss_fn = TASK_LOSS_FNS[task]()
    if CUDA:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    # get trainable params 
    # TODO: should this be strict like suggested in 
    #       https://github.com/pytorch/pytorch/issues/679 and not made magic?)
    trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())

    opt = Adam(trainable_model_params, lr=lr, weight_decay=weight_decay)

    def step(xs, ys, i_step=None):
        xs = Variable(xs, requires_grad=False)
        # NOTE: if task is 'autoencode', ys should be desired output image
        ys = Variable(ys, requires_grad=False)
        if CUDA:
            xs, ys = xs.cuda(), ys.cuda()

        model.train()
        opt.zero_grad()
        yhl = model(xs)
        loss = loss_fn(yhl, ys)

        train_loss = loss
        if training_penalty:
            pen = training_penalty(model=model, inputs=xs, outputs=yhl)
            train_loss = train_loss + pen

        train_loss.backward()
        opt.step()
        loss = var_to_numpy(loss)[0]
        train_loss = var_to_numpy(train_loss)[0]
        if i_step is not None:
            print('(step {:d}) loss: {:.3f}, train_loss: {:.3f}'\
                  .format(i_step, loss, train_loss), end='\r')


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
            loss = var_to_numpy(loss_fn(yhl, ys))

            if task is 'classify':
                _, yh = yhl.max(1)
                acc = mean(var_to_numpy(yh == ys))
            else:
                acc = -1.

            losses.append(loss)
            accs.append(acc)

        loss = mean(losses)
        acc = mean(accs)
        print('({:s} step {:d}) acc: {:.3f}, loss: {:.3f}'\
                .format(split, i_step, acc, loss))
        return loss, acc

    i_step = 0
    results = {'train': {'loss': {}, 'acc': {}},
               'test': {'loss': {}, 'acc': {}}}
    rtrain, rtest = results['train'], results['test']
    for i_epoch in range(n_epoch):
        rtrain['loss'][i_epoch], rtrain['acc'][i_epoch] = evaluate(train_data_loader, 'train', i_step)
        rtest['loss'][i_epoch], rtest['acc'][i_epoch] = evaluate(test_data_loader, 'test', i_step)
        for xs, ys in train_data_loader:
            step(xs, ys, i_step=i_step)
            i_step += 1
    return model, results, data_loaders


def _flatten(im):
    return im.view(-1)


def _channel_dim(im):
    return im.unsqueeze(0)

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset

# TODO (maybe): add a few common transforms as string options
# QUICK_TRANSFORMS = {'flatten': _flatten, 'channel_dim': channel_dim}

from numbers import Number
from pyt.training import split_inds

def quick_mnist(data_dir, 
                batch_size, 
                transform=ToTensor(), 
                p_splits=None, 
                balanced=False, 
                seed=None,
                num_workers=0,
                pin_memory=False):
    os.makedirs(data_dir, exist_ok=True)
    download = not (os.path.exists(os.path.join(data_dir, 'processed', 'training.pt'))
                    and os.path.exists(os.path.join(data_dir, 'processed', 'test.pt')))

    train_dataset = MNIST(data_dir, 
                    download=download,
                    transform=transform,
                    train=True)

    test_dataset = MNIST(data_dir, 
                    download=False,
                    transform=transform,
                    train=False)

    if isinstance(p_splits, Number):
        p_splits = {'train': p_splits, 'val': 1. - splits}
    elif p_splits is None:
        p_splits = {'train': 1.}

    sinds = split_inds(len(train_dataset), 
                         p_splits, 
                         balanced=balanced, 
                         labels=train_dataset.train_labels.numpy().astype(int),
                         seed=seed)

    samplers = {split: SubsetRandomSampler(inds) 
                for split, inds in sinds.items()}

    data_loaders = {split: DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     sampler=sam,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)
                    for split, sam in samplers.items()}
    data_loaders['test'] = DataLoader(test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
    return data_loaders


if __name__ == '__main__':
    from pyt.core_modules import MLP, Identity, Swish
    from pyt.util import vlt, var_to_numpy
    import os
    import torch

    # example use case

    # mnist path
    DATA_DIR = os.path.join(os.getenv('HOME'), 'datasets', 'mnist')

    # build mnist-compatible model
    model = MLP(784, 10, [128]*3, act_fn=Swish())

    # what are we testing? (currently supports 'classify' and 'autoencode')
    task = 'classify'

    # training kwargs has opt and batching kwargs in it atm 
    # TODO: should these be separated?
    training_kwargs = {'lr': 1e-3, 
                       'weight_decay': 1e-4,
                       'n_epoch': 6}


    # EXAMPLE OF PASSING CUSTOM DATA TRANSFORM
    # how would we like to transform the data for our model?
    # NOTE: not using lambdas so we can serialize
    toTensor = ToTensor()
    def custom_transform(pil_image):
        return toTensor(pil_image).view(-1).float()


    # EXAMPLE OF QUICKLY ADDING A CUSTOM PENALTY.
    # penalty can be anything so long as can accept
    # (and possible ignore) model, inputs, and outputs,
    # and return a scalar Variable to add to loss
    # during training
    def custom_l1_act_penalty(model: nn.Module,\
                              inputs: Variable,
                              outputs: Variable=None,
                              weight: float=1.)\
                             -> Variable:
        """add sparisty penalty to all hidden activations"""
        pen = Variable(torch.zeros(1))
        output = inputs
        for lin, norm in zip(model.linear, model.normed):
            output = model.act_fn(norm(lin(output)))
            pen += output.norm(1)
        return pen.mean() * 1e-5


    # that's it!
    model, results, data_loaders = test_over_mnist(model=model,
                                                   data_dir=DATA_DIR, 
                                                   task=task, 
                                                   transform=custom_transform,
                                                   training_kwargs=training_kwargs,  # optional
                                                   training_penalty=custom_l1_act_penalty)  # optional
