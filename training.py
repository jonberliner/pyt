import numpy as np
from numbers import Number

def split_inds(inds, p_splits, balanced=False, labels=None, seed=None):
    rng = np.random.RandomState(seed)
    assert balanced is False, 'balanced not yes implemented'

    if type(inds) == int:
        inds = np.arange(inds)


    assert len(inds.shape) == 1, 'inds must be a vector of ints'

    if isinstance(p_splits, Number):
        p_splits = dict(train=p_splits, 
                        test=1.-p_splits)
        ps = np.array(list(p_splits.values()))
        assert ((ps > 0.) and (ps <= 1.)).all(), 'all p_splits passed must be in (0., 1.]'
        assert ps.sum() <= 1., 'all p_splits must sum to <= 1.'

    num_inds = len(inds)
    split_inds = dict()

    if balanced:
        assert labels is not None
        assert len(labels) == len(inds)
    else:
        pool = rng.permutation(inds)
        i_start = 0
        for split in p_splits:
            n_split = min(int(p_splits[split] * num_inds), 1)  # no empty splits
            i_end = i_start + n_split
            split_inds[split] = inds[i_start:i_end]
            i_start = i_end
        # assign dangling to test
        if i_start < num_inds:
            split_inds['test'] = np.concatenate([split_inds['test'], 
                                                 inds[i_start:]], 0)
    return split_inds


class Batcher(object):
    def __init__(self, data, batch_size, drop_last=False, shuffle=True, seed=None):
        if type(data) == int:
            data = np.arange(data)
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.data = data
        self.num_data = len(self.data)
        self.batch_size = batch_size
        assert self.num_data >= self.batch_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.i_pool = np.arange(self.num_data)

        # will start proper on first call
        self._new_epoch()

        self.curr_pool_inds = None

    def _new_epoch(self):
        if self.shuffle:
            self.i_pool = self.rng.permutation(self.i_pool)
        self._i_start = 0
        self._i_end = self.batch_size

    def __call__(self):
        if self.end_of_epoch():
            self._new_epoch()

        _inds = self.i_pool[self._i_start:self._i_end]
        output = self.data[_inds]

        self.curr_pool_inds = (self._i_start, self._i_end)

        # FIXME: this tells you what inds are next, not the ones from the current batch
        self._increment_inds()
        return output

    def end_of_epoch(self):
        if self.drop_last:  # drop last batch if not same batch size
            eoe = self._i_end - self._i_start < self.batch_size
        else:
            eoe = self._i_start == self._i_end
        return eoe

    def _increment_inds(self):
        self._i_start = self._i_end
        self._i_end = min(self._i_end + self.batch_size, self.num_data)



