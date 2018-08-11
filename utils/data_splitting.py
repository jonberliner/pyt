import os

import numpy as np
from numpy.random import RandomState


def split_indices(inds,
                  p_splits,
                  balanced_labels=False,
                  labels=None,
                  seed=None):
    """
    take in indices are cut up by probs specified in p_splits,
    which options for balancing label proportions across splits

    Args:
        inds (int or 1d array_like): indices to split between splits.
            if int passed, will use indices np.arange(int)
        p_splits (float in (0., 1.] or dict)): percentage of indices
            to use for each split.
            if float passed < 1., will use
                {'train': p_splits, 'val': 1.-p_splits}.
            If float passed == 1., will use
                {'train': p_splits}
        balanced_labels (bool; default False): ensure equal label proportions
            across splits.  if true, must pass labels for each index
        labels (1d array_like): list of labels for each index.
            Must be same length as inds
        seed (int; default None): seed for data splitting.  If None, uses
            a random seed
    Returns:
        {string: [int]}: dict of indices for each split
    """

    # ensure inds is 1d nparray
    if isinstance(inds, (int, np.integer)):
        inds = np.arange(inds)
    else:
        inds = np.array(inds)
    assert inds.ndim == 1, 'inds must be a 1d array_like or an int'

    p_splits = process_p_splits(p_splits)

    rng = RandomState(seed)
    if balanced_labels:
        assert labels is not None
        splits = _balanced_label_split_inds(
                inds=inds,
                p_splits=p_splits,
                rng=rng,
                labels=labels)
    else:
        splits = _split_inds(
                inds=inds,
                p_splits=p_splits,
                rng=rng)

    return splits


def process_p_splits(p_splits, ensure_valid=True):
    """
    ensure p_splits is a dict with floats for values.
    if ensure_valid == True, check that all split percentages in (0., 1.],
    and that all split percentages sum to <= 1.
    """
    if isinstance(p_splits, (float, np.floating)):
        assert p_splits > 0. and p_splits <= 1.,\
            'if p_splits is float, must be in (0., 1.]'
        p_splits = {'train': p_splits}
        if p_splits['train'] < 1.:
            p_splits['val'] = 1. - p_splits
    else:
        assert isinstance(p_splits, dict)
        for p_split in p_splits.values():
            assert isinstance(p_split, (float, np.floating))

    if ensure_valid:
        # ensure valid probabilities for splits
        ps = np.array(list(p_splits.values()))
        assert (ps > 0.).all() and (ps <= 1.).all(),\
            'all p_splits passed must be in (0., 1.]'
        assert ps.sum() <= 1., 'all p_splits must sum to <= 1.'
    return p_splits


def _balanced_label_split_inds(inds, p_splits, rng, labels):
    """split inds with equal label proportions across splits"""
    assert len(labels) == len(inds)

    split_inds = {split: list() for split in p_splits}
    for lab in np.unique(labels):
        # get label indices
        lab_inds = np.where(labels == lab)[0]
        num_lab_inds = len(lab_inds)

        # assign label inds to splits
        shuffled_lab_inds = rng.permutation(lab_inds)
        i_start = 0
        for split, p_split in p_splits.items():
            # no empty splits
            n_split = np.ceil(p_split * num_lab_inds).astype(int)
            i_end = i_start + n_split
            split_inds[split].append(shuffled_lab_inds[i_start:i_end])
            i_start = i_end

    split_inds = {split: np.concatenate(sinds, 0)
                  for split, sinds in split_inds.items()}
    return split_inds


def _split_inds(inds, p_splits, rng):
    """split inds randomly across splits"""
    num_inds = len(inds)
    shuffled_inds = rng.permutation(inds)
    split_inds = {split: None for split in p_splits}
    i_start = 0
    for split, p_split in p_splits.items():
        # no empty splits
        n_split = np.ceil(p_split * num_inds).astype(int)
        i_end = i_start + n_split
        split_inds[split] = shuffled_inds[i_start:i_end]
        i_start = i_end
    return split_inds
