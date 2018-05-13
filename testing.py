from torch import is_tensor
from torch.utils.data import DataLoader
import torchvision.datasets as tvdatasets
from torch import nn
from torch.optim import Adam
from numpy import mean
from numpy import array as npa
import os
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from numbers import Number
from pyt.training import split_inds

from pyt.util import CUDA, var_to_numpy

TASK_LOSS_FNS = {'autoencode': nn.BCEWithLogitsLoss,
                 'classify': nn.CrossEntropyLoss,
                 'regress': nn.MSELoss}

TASKS = list(TASK_LOSS_FNS.keys())


def test_over_dataset(model,
                      dataset,
                      data_dir,
                      task,
                      training_kwargs={},
                      transform=ToTensor(),
                      data_seed=None,
                      training_penalty=None,
                      p_splits={'train': 0.8, 'val': 0.2},
                      balanced=True,
                      include_test_data_loader=True):
    """quickly train a model over mnist
    Args:
        model (nn.Module): pytorch model to optimize
        dataset (str): torchvision dataset to train over
        data_dir (str): dir to (possibly download and) find mnist data
        task (str): type of mnist task.  must be in {:s}
        flatten (bool): flatten data (bs x 784) or not? (bs x 28 x 28)
        channel_dim (bool): add channel_dim? (bs x {ch} x ...)
        training_kwargs (dict; optional): kwargs for training
        custom_pen (nn.Module; optional): takes in custom penalties.
            takes in model, inputs, and outputs (all optional))"""
    assert task in TASKS, f'task must be in {str(TASKS)}'
    n_epoch = training_kwargs.get('n_epoch', 10)
    batch_size = training_kwargs.get('batch_size', 64)
    lr = training_kwargs.get('lr', 1e-3)
    weight_decay = training_kwargs.get('weight_decay', 0.)

    data_loaders = quick_dataset(dataset,
                                 data_dir,
                                 batch_size,
                                 transform=transform,
                                 balanced=balanced,
                                 seed=data_seed,
                                 p_splits=p_splits,
                                 include_test_data_loader=include_test_data_loader)  # noqa:E501

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loss_fn = TASK_LOSS_FNS[task]()

    model = model.to(device)
    # TODO: figure out if this is supposed to go to gpu
    # loss_fn = loss_fn.to(device)

    # get trainable params
    # TODO: should this be strict like suggested in
    #       https://github.com/pytorch/pytorch/issues/679 and not made magic?)
    trainable_model_params = filter(lambda p: p.requires_grad,
                                    model.parameters())

    opt = Adam(trainable_model_params,
               lr=lr,
               weight_decay=weight_decay)

    def step(xs, ys, i_step=None):
        xs, ys = xs.to(device), ys.to(device)

        model.train()
        opt.zero_grad()

        yhl = model(xs)
        loss = loss_fn(yhl, ys)
        if task == 'classify':
            _, yhat = yhl.max(1)
            acc = (ys == yhat).float().mean()

        train_loss = loss
        if training_penalty:
            pen = training_penalty(model=model, inputs=xs, outputs=yhl)
            train_loss = train_loss + pen

        train_loss.backward()
        opt.step()
        if i_step is not None:
            message = (f'(step {i_step}) loss: {loss.item():.3f}'
                       f', train_loss: {train_loss.item():.3f}')
            if task == 'classify':
                message = message + f', acc: {acc.item():.3f}'
            print(message, end='\r')

    def evaluate(data_loader, split, i_step):
        n_sam = 0
        model.eval()
        loss, n_correct = 0., 0.
        n_batch = len(data_loader)
        with torch.no_grad():
            for i_batch, xy in enumerate(data_loader):
                xs, ys = xy
                xs, ys = xs.to(device), ys.to(device)
                batch_size = xs.shape[0]
                n_sam += batch_size
                print(f'testing split {split} (batch {i_batch} of {n_batch})',
                      end='\r')
                yhl = model(xs)
                loss += (loss_fn(yhl, ys) * batch_size).item()

                if task is 'classify':
                    _, yh = yhl.max(1)
                    n_correct += (yh == ys).sum().item()
                else:
                    acc = -1.

        loss = loss / n_sam
        acc = n_correct / n_sam
        print('({:s} step {:d}) acc: {:.3f}, loss: {:.3f}'
              .format(split, i_step, acc, loss))
        return loss, acc

    i_step = 0
    results = {split: {'loss': {}, 'acc': {}} for split in data_loaders}
    for i_epoch in range(n_epoch):
        for split_name, split_res in results.items():
            split_res['loss'][i_epoch] = evaluate(data_loaders[split_name],
                                                  split_name,
                                                  i_step)

        for xs, ys in data_loaders['train']:
            step(xs, ys, i_step=i_step)
            i_step += 1
    return model, results, data_loaders


def _flatten(im):
    return im.view(-1)


def _channel_dim(im):
    return im.unsqueeze(0)


# TODO (maybe): add a few common transforms as string options
# QUICK_TRANSFORMS = {'flatten': _flatten, 'channel_dim': channel_dim}

# def quick_mnist(data_dir,
DATASETS = ['SVHN', 'FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100']
lower_DATASETS = [ds.lower() for ds in DATASETS]


def quick_dataset(dataset_name,
                  data_dir,
                  batch_size=32,
                  transform=ToTensor(),
                  p_splits=None,
                  balanced=False,
                  seed=None,
                  num_workers=0,
                  pin_memory=False,
                  include_test_data_loader=True):
    try:
        index_ds = lower_DATASETS.index(dataset_name.lower())
    except ValueError:
        raise ValueError(f'{dataset_name.lower()} not in {lower_DATASETS}')
    dataset_name = DATASETS[index_ds]
    dataset = getattr(tvdatasets, dataset_name)

    data_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    download = not (os.path.exists(
                        os.path.join(data_dir, 'processed', 'training.pt'))
                    and os.path.exists(
                            os.path.join(data_dir, 'processed', 'test.pt')))

    train_dataset = dataset(data_dir,
                            download=download,
                            transform=transform,
                            train=True)

    if isinstance(p_splits, Number):
        p_splits = {'train': p_splits,
                    'val': 1. - p_splits}
    elif p_splits is None:
        p_splits = {'train': 1.}

    labels = train_dataset.train_labels
    if is_tensor(labels):
        labels = labels.numpy().astype(int)
    elif type(labels) is list:
        labels = npa(labels).astype(int)

    sinds = split_inds(len(train_dataset),
                       p_splits,
                       balanced=balanced,
                       labels=labels,
                       seed=seed)

    samplers = {split: SubsetRandomSampler(inds)
                for split, inds in sinds.items()}

    data_loaders = {split: DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=sam,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
                    for split, sam in samplers.items()}

    if include_test_data_loader:
        test_dataset = dataset(data_dir,
                               download=False,
                               transform=transform,
                               train=False)
        data_loaders['test'] = DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)
    return data_loaders


if __name__ == '__main__':
    from pyt.modules import MLP, Swish
    import torch
    from torchvision.transforms import Compose

    # EXAMPLE USE CASE

    # datasets directory path
    DATA_DIR = os.path.join(os.getenv('HOME'), 'datasets')
    DATASET = 'FashionMNIST'  # will be in $DATA_DIR/$DATASET

    # build mnist-compatible model
    model = MLP(784, 10, [128]*3,
                p_drop=0.2, dropout_kind='gaussian',
                act_fn=Swish(), norm=None)

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
    def custom_transform(pil_image):
        return Compose([ToTensor(),
                        lambda im: im.view(-1).float()])

    # EXAMPLE OF QUICKLY ADDING A CUSTOM PENALTY.
    # penalty can be anything so long as can accept
    # (and possible ignore) model, inputs, and outputs,
    # and return a scalar Variable to add to loss
    # during training
    def custom_l1_act_penalty(model: nn.Module,
                              inputs: torch.Tensor,
                              outputs: torch.Tensor=None,
                              weight: float=1.)\
            -> torch.Tensor:
        """add sparisty penalty to all hidden activations"""
        pen = torch.zeros(1)
        output = inputs
        for lin, norm in zip(model.linear, model.normed):
            output = model.act_fn(norm(lin(output)))
            pen += output.norm(1)
        return pen.mean() * 1e-5

    # that's it!
    model, results, data_loaders =\
        test_over_dataset(model=model,
                          task=task,
                          dataset=DATASET,
                          data_dir=DATA_DIR,
                          p_splits={'train': 1.},
                          include_test_data_loader=True,
                          transform=custom_transform,
                          training_kwargs=training_kwargs,  # optional
                          training_penalty=custom_l1_act_penalty)  # optional
