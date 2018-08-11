import os

from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import is_tensor
from torch.utils.data.sampler import SubsetRandomSampler

from numpy import array as np_array

from pyt.utils.data_splitting import split_indices

DATASETS = ['SVHN', 'FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100']
lower_DATASETS = [ds.lower() for ds in DATASETS]


def quick_dataset(dataset_name,
                  data_dir,
                  batch_size=32,
                  transform=ToTensor(),
                  p_splits=1.,
                  balanced_labels=False,
                  seed=None,
                  num_workers=0,
                  pin_memory=False,
                  include_test_data_loader=True):
    try:
        index_ds = lower_DATASETS.index(dataset_name.lower())
    except ValueError:
        raise ValueError(f'{dataset_name.lower()} not in {lower_DATASETS}')
    dataset_name = DATASETS[index_ds]
    dataset = getattr(tv_datasets, dataset_name)

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

    labels = train_dataset.train_labels
    if is_tensor(labels):
        labels = labels.numpy().astype(int)
    labels = np_array(labels)

    sinds = split_indices(inds=len(train_dataset),
                          p_splits=p_splits,
                          balanced_labels=balanced_labels,
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
