import warnings

import torch


def first_n_classes_subset(dataset, first_n_classes):
    targets = torch.tensor(dataset.targets)
    idx = targets < first_n_classes
    dataset.targets = targets[idx].tolist()
    dataset.data = dataset.data[idx]
    return dataset


def n_samples_per_class(dataset, n_samples_per_class, num_of_classes, exact=False):
    curr_samples_per_class = torch.ones(num_of_classes, dtype=torch.int32)
    idx = []
    for i, target in enumerate(dataset.targets):
        if torch.all(curr_samples_per_class == n_samples_per_class):
            break
        elif curr_samples_per_class[target] < n_samples_per_class:
            curr_samples_per_class[target] += 1
            idx.append(i)

    if exact:
        assert torch.all(curr_samples_per_class == n_samples_per_class), f"Could not find {n_samples_per_class} samples for each class (samples_per_class: {curr_samples_per_class})"
    elif not torch.all(curr_samples_per_class == n_samples_per_class):
        warnings.warn(f"Warning: Could not find {n_samples_per_class} samples for each class (samples_per_class: {curr_samples_per_class})")

    dataset.targets = [dataset.targets[i] for i in idx]
    dataset.data = dataset.data[idx]
    return dataset