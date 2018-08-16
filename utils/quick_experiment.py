import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import ToTensor

from pyt.utils.quick_data import quick_dataset

use_cuda = torch.cuda.is_available()

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
                     include_test_data_loader=True,
                     evaluate_before_training=False,
                     epochs_per_eval=1,
                     print_training_loss_every=100):
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

    device = 'cuda:0' if use_cuda else 'cpu'
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

    def step(xs, ys):
        xs, ys = xs.to(device), ys.to(device)

        model.train()
        opt.zero_grad()

        yhl = model(xs)
        loss = loss_fn(yhl, ys)
        if task == 'classify':
            _, yhat = yhl.max(1)
            acc = (ys == yhat).float().mean()
        else:
            acc = -1

        train_loss = loss
        if training_penalty:
            pen = training_penalty(model=model, inputs=xs, outputs=yhl)
            train_loss = train_loss + pen

        train_loss.backward()
        opt.step()
        return loss.item(), acc.item()

    def evaluate(data_loader, split, i_epoch, i_step):
        model.eval()
        n_batch = len(data_loader)
        loss, n_correct = 0., 0.
        n_sam = 0
        with torch.no_grad():
            for i_batch, xy in enumerate(data_loader):
                xs, ys = xy
                xs, ys = xs.to(device), ys.to(device)
                batch_size = xs.shape[0]
                n_sam += batch_size
                print(f'testing split {split} (batch {i_batch} of {n_batch})',
                      end='\r')
                yhl = model(xs)
                loss += loss_fn(yhl, ys).item() * batch_size

                if task is 'classify':
                    _, yh = yhl.max(1)
                    n_correct += (yh == ys).sum().item()
                else:
                    acc = -1.

        loss = loss / n_sam
        acc = n_correct / n_sam
        print('EVAL: (split: {:s}, epoch {:d}, step: {:d}) loss: {:.3f}, acc: {:.3f}'
              .format(split, i_epoch, i_step, loss, acc))
        return loss, acc

    i_step = 0
    results = {split: {'loss': {}, 'acc': {}} for split in data_loaders}

    running_loss, running_acc = None, None
    if evaluate_before_training:
        for split_name, split_res in results.items():
            split_res['loss'][i_epoch] = evaluate(data_loaders[split_name],
                                              split_name,
                                              i_epoch,
                                              i_step)
    for i_epoch in range(n_epoch):
        for xs, ys in data_loaders['train']:
            loss, acc = step(xs, ys)

            # update running loss
            if i_step == 0:
                running_loss, running_acc = loss, acc
            else:
                running_loss = running_loss * 0.99 + loss * 0.01
                running_acc = running_acc * 0.99 + acc * 0.01

            if i_step % print_training_loss_every == 0:
                message = f'(step {i_step}) loss: {running_loss:.3f}'
                if task == 'classify':
                    message = message + f', acc: {running_acc:.3f}'
                print(message)
            i_step += 1

        if i_epoch % epochs_per_eval == 0:
            for split_name, split_res in results.items():
                split_res['loss'][i_epoch] = evaluate(data_loaders[split_name],
                                                      split_name,
                                                      i_epoch,
                                                      i_step)
    return model, results, data_loaders
