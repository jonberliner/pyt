from torch import nn
from torch.optim import Adam
from torchvision.transforms import ToTensor

from pyt.utils.quick_data import quick_dataset

TASK_LOSS_FNS = {'autoencode': nn.BCEWithLogitsLoss,
                 'classify': nn.CrossEntropyLoss,
                 'regress': nn.MSELoss}

TASKS = list(TASK_LOSS_FNS.keys())


def quick_experiment(model,
                     dataset,
                     data_dir,
                     task,
                     training_kwargs={},
                     transform=ToTensor(),
                     data_seed=None,
                     training_penalty=None,
                     p_splits={'train': 0.8, 'val': 0.2},
                     balanced_labels=True,
                     include_test_data_loader=True):
    """
    quickly train a pytorch model over a dataset in an autoencoding,
    classification, or regression task

    Args:
        model (nn.Module): pytorch model to optimize.
            all tasks expect model to output logits
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

    # TODO: remove quick_dataset dependency.  abstract to train_dataloaders
    #       and eval_dataloaders
    data_loaders = quick_dataset(dataset,
                                 data_dir,
                                 batch_size,
                                 transform=transform,
                                 balanced_labels=balanced_labels,
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


if __name__ == '__main__':
    import os
    import torch
    from torchvision.transforms import Compose

    # EXAMPLE USE CASE

    # datasets directory path
    DATA_DIR = os.path.join(os.getenv('HOME'), 'datasets')
    DATASET = 'MNIST'  # will be in $DATA_DIR/$DATASET

    # build mnist-compatible model
    class SimpleMNISTMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(784, 256)
            self.b1 = nn.BatchNorm1d(256)
            self.relu = nn.ReLU()
            self.l2 = nn.Linear(256, 10)

            self.hidden = None

        def forward(self, inputs):
            self.hidden = self.relu(self.b1(self.l1(inputs)))
            return self.l2(self.hidden)

    model = SimpleMNISTMLP()

    # what are we testing? (currently supports 'classify' and 'autoencode')
    task = 'classify'

    # training kwargs has opt and batching kwargs in it atm
    # TODO: should these be separated?
    training_kwargs = {'lr': 1e-2,
                       'n_epoch': 6,
                       'batch_size': 128}

    # EXAMPLE OF PASSING CUSTOM DATA TRANSFORM
    # how would we like to transform the data for our model?
    # NOTE: not using lambdas so we can serialize
    custom_transform = Compose([ToTensor(),
                                lambda im: im.view(-1).float()])

    # EXAMPLE OF QUICKLY ADDING A CUSTOM PENALTY.
    # penalty can be anything so long as can accept
    # (and possible ignore) model, inputs, and outputs,
    # and return a scalar Variable to add to loss
    # during training
    def custom_l1_act_penalty(model: nn.Module,
                              inputs: torch.Tensor,
                              outputs: torch.Tensor=None) -> torch.Tensor:
        """add sparisty penalty to all hidden activations"""
        if model.hidden is None:
            pen = torch.tensor(0.)
        else:
            pen = model.hidden.norm(1).mean()
        return pen

    def custom_penalty(model: nn.Module,
                       inputs: torch.Tensor,
                       outputs: torch.Tensor=None,
                       weight: float=1.) -> torch.Tensor:
        l1_act_pen = custom_l1_act_penalty(model, inputs, outputs)
        l2 = torch.tensor(0.)
        for param in model.parameters():
            l2 += param.norm(2)
        return l1_act_pen * 1e-4 + l2 * 1e-4

    # that's it!
    model, results, data_loaders =\
        quick_experiment(model=model,
                         task=task,
                         dataset=DATASET,
                         data_dir=DATA_DIR,
                         p_splits={'train': 1.},
                         include_test_data_loader=True,
                         transform=custom_transform,
                         training_kwargs=training_kwargs,  # optional
                         training_penalty=custom_penalty)  # optional
