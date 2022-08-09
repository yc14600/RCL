from pathlib import Path
from typing import Optional, Sequence, Union, Any
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    ToPILImage,
    Compose,
    Normalize,
    RandomRotation,
)
import numpy as np
import torchvision.transforms as transforms
from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheDataset
_default_mnist_train_transform = Compose(
    [ToTensor(), Normalize((0,), (1,))]
)

_default_mnist_eval_transform = Compose(
    [ToTensor(), Normalize((0,), (1,))]
)

def SplitMNIST(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    train_transform: Optional[Any] = _default_mnist_train_transform,
    eval_transform: Optional[Any] = _default_mnist_eval_transform,
    dataset_root: Union[str, Path] = None
):
    """
    Creates a CL benchmark using the MNIST dataset.
    If the dataset is not present in the computer, this method will
    automatically download and store it.
    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.
    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).
    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.
    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".
    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.
    :returns: A properly initialized :class:`NCScenario` instance.
    """

    mnist_train = MNIST(root=dataset_root, train=True, download=True)

    mnist_test = MNIST(root=dataset_root, train=False, download=True)

    if return_task_id:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )


__all__ = ["SplitMNIST"]