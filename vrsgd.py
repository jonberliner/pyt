import torch
from copy import deepcopy


# helper functions
def _zero_model_params(model):
    """zero all parameters of a model (BE CAREFUL!)"""
    [param.data.zero_() for param in model.parameters()]
    return model


def _get_loss(model, loss_fn, inputs, outputs):
    yhat = model(inputs)
    return loss_fn(yhat, outputs)


def _get_gradients(model, loss_fn, inputs, outputs):
    model.zero_grad()
    yhat = model(inputs)
    loss_fn(yhat, outputs).backward()
    return (param.grad for param in model.parameters())


def _get_full_loss(model, loss_fn, data_loader):
    """get the average gradient over all data in data_loader"""
    # NOTE: only works w standard torchvision dataset organization atm
    full_loss = 0.
    model.zero_grad()
    for batch_x, batch_y in data_loader:
        n_batch = batch_x.shape[0]
        full_loss += _get_loss(model, loss_fn, batch_x, batch_y) * n_batch
    return full_loss / len(data_loader.dataset)


def _get_full_grads(model, loss_fn, data_loader):
    """get the average gradient over all data in data_loader"""
    # NOTE: only works w standard torchvision dataset organization atm
    full_grads = None
    model.zero_grad()
    for batch_x, batch_y in data_loader:
        batch_grads = _get_gradients(model, loss_fn, batch_x, batch_y)
        if full_grads is None:
            full_grads = batch_grads
        else:
            for fg, bg in zip(full_grads, batch_grads):
                fg += bg
        # average
        for grad in full_grads:
            grad /= len(data_loader.dataset)
    return full_grads


def _sum_a_nb_c(a, b, c):
    return a - b + c


def vrsgd(n_epoch,
          steps_per_epoch,
          model,
          loss_fn,
          data_loader,
          initial_lr=1e-1,
          minimum_lr_divisor=1e-5,
          add_last_step_to_snapshot=True):
    # dataset stuff
    num_samples = len(data_loader.dataset)
    inputs = data_loader.dataset.train_data
    outputs = data_loader.dataset.train_labels

    # OUTER LOOP
    avg_snapshot_model = _zero_model_params(deepcopy(model))
    snapshot_model = deepcopy(model)
    current_step_model = model
    for epoch in range(n_epoch):
        lr = initial_lr / max(minimum_lr_divisor, 2. / (epoch + 1.))
        # compute full gradient at from last snapshot
        snapshot_full_grads = _get_full_grads(snapshot_model,
                                              loss_fn,
                                              data_loader)

        # INNER LOOP
        # for saving mean of params in inner loop
        _snapshot_model = deepcopy(snapshot_model)
        [param.data.zero_() for param in _snapshot_model.parameters()]

        perm = torch.randperm(num_samples)[:steps_per_epoch]
        for i_step, index in enumerate(perm):
            # gradients from the current model over this piece of data
            current_step_grads = _get_gradients(current_step_model,
                                                loss_fn,
                                                inputs[index:index+1],
                                                outputs[index:index+1])
            # gradients from the snapshot over this piece of data
            snapshot_step_grads = _get_gradients(snapshot_model,
                                                 loss_fn,
                                                 inputs[index:index+1],
                                                 outputs[index:index+1])

            current_update_grads =\
                (grad for grad in map(_sum_a_nb_c,
                                      current_step_grads,
                                      snapshot_step_grads,
                                      snapshot_full_grads))

            # inner loop update
            for _snapshot_param, current_step_param, current_step_grad in\
                zip(_snapshot_model.parameters(),
                    current_step_model.parameters(),
                    current_update_grads):
                # update for next step in inner loop
                current_step_param.data -= lr * current_step_grad
                # save accumulated params for outer-loop update
                if add_last_step_to_snapshot or i_step < steps_per_epoch:
                    _snapshot_param.data += current_step_param

        # update snapshot model
        for param in _snapshot_model.parameters():
            param.data /= steps_per_epoch
        snapshot_model = _snapshot_model

        # update average model
        for sp, asp in zip(snapshot_model.parameters(),
                           avg_snapshot_model.parameters()):
            asp.data += sp

    for param in avg_snapshot_model.parameters():
        param.data /= n_epoch

    # return either last snapshot or average snapshot based
    # on which has lower full-dataset loss
    avg_snapshot_loss = _get_full_loss(avg_snapshot_model,
                                       loss_fn,
                                       data_loader)
    last_snapshot_loss = _get_full_loss(snapshot_model,
                                        loss_fn,
                                        data_loader)
    if avg_snapshot_loss < last_snapshot_loss:
        return avg_snapshot_model
    else:
        return snapshot_model
