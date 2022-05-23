import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader, Sampler
from PIL import Image

from active_learning_utils import get_top_n
import collections
from alternative_mnist import ImbalancedMNIST, TwoMoons, ImbalancedFashionMNIST


class WeightedTwoMoons(TwoMoons):
    """
    Dataloader for TwoMooons that also returns a weight for each sample.
    The weight is

    = \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right)

    from the equation:

    \rtil_a = \frac{1}{MN} \sum_{m=1}^M \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right) L_{i_m}

    Multiplying the loss for each data point by this weight, and dividing by M (Number of labelled points)
    and N (total number of points in the dataset including unlabelled) gives you an unbiased estimator of
    the empirical risk.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proposal_mass = torch.zeros_like(self.targets).float()
        self.average_proposal_mass = torch.zeros_like(self.targets).double()
        self.normalization = torch.zeros_like(self.targets).double()
        self.sample_order = torch.zeros_like(self.targets)
        self.weights = torch.zeros_like(self.targets).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, weight = (
            self.data[index],
            int(self.targets[index]),
            self.weights[index],
        )

        return img, target, weight


class WeightedMNIST(ImbalancedMNIST):
    """
    Dataloader for MNIST that also returns a weight for each sample.
    The weight is

    = \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right)

    from the equation:

    \rtil_a = \frac{1}{MN} \sum_{m=1}^M \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right) L_{i_m}

    Multiplying the loss for each data point by this weight, and dividing by M (Number of labelled points)
    and N (total number of points in the dataset including unlabelled) gives you an unbiased estimator of
    the empirical risk.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proposal_mass = torch.zeros_like(self.targets).float()
        self.average_proposal_mass = torch.zeros_like(self.targets).double()
        self.normalization = torch.zeros_like(self.targets).double()
        self.sample_order = torch.zeros_like(self.targets)
        self.weights = torch.zeros_like(self.targets).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, weight = (
            self.data[index],
            int(self.targets[index]),
            self.weights[index],
        )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, weight


class WeightedFashionMNIST(ImbalancedFashionMNIST):
    """
    Dataloader for MNIST that also returns a weight for each sample.
    The weight is

    = \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right)

    from the equation:

    \rtil_a = \frac{1}{MN} \sum_{m=1}^M \left(\frac{1}{q(i_m;i_{1:m-1},D)}+M-m\right) L_{i_m}

    Multiplying the loss for each data point by this weight, and dividing by M (Number of labelled points)
    and N (total number of points in the dataset including unlabelled) gives you an unbiased estimator of
    the empirical risk.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proposal_mass = torch.zeros_like(self.targets).float()
        self.average_proposal_mass = torch.zeros_like(self.targets).double()
        self.normalization = torch.zeros_like(self.targets).double()
        self.sample_order = torch.zeros_like(self.targets)
        self.weights = torch.zeros_like(self.targets).float()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target, weight = (
            self.data[index],
            int(self.targets[index]),
            self.weights[index],
        )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, weight


def get_subset_base_indices(dataset, indices):
    return [int(dataset.indices[index]) for index in indices]


def sample_proportionally(probability_masses, active_learning_hypers):
    num_to_acquire = active_learning_hypers["num_to_acquire"]
    assert len(probability_masses) >= num_to_acquire
    sampled_idxs = torch.multinomial(probability_masses, num_to_acquire)
    return sampled_idxs, probability_masses[sampled_idxs]


def sample_softmax(probability_masses, active_learning_hypers):
    num_to_acquire = active_learning_hypers["num_to_acquire"]
    temperature = float(active_learning_hypers["temperature"])
    assert len(probability_masses) >= num_to_acquire
    probability_masses = torch.nn.functional.softmax(
        temperature * probability_masses, dim=0
    )
    sampled_idxs = torch.multinomial(probability_masses, num_to_acquire)
    return sampled_idxs, probability_masses[sampled_idxs]


class ActiveLearningData:
    def __init__(self, dataset, active_learning_hypers):
        super().__init__()
        self.dataset = dataset
        self.total_num_points = len(self.dataset)
        self.num_initial_points = active_learning_hypers["starting_points"]
        self.weighting_scheme = active_learning_hypers["weighting_scheme"]
        self.active_learning_hypers = active_learning_hypers

        # At the beginning, we have acquired no points
        # the acquisition mask is w.r.t. the training data (not validation)
        self.acquisition_mask = np.full((len(dataset),), False)
        self.num_acquired_points = 0

        self.active_dataset = Subset(self.dataset, None)
        self.available_dataset = Subset(self.dataset, None)

        # # Now we randomly select num_initial_points uniformly
        self.num_acquired_points = self.num_initial_points
        self._update_indices()

        for initial_idx in range(self.num_initial_points):
            scores = torch.ones(len(self.available_dataset))
            self.acquire_points_and_update_weights(scores, self.active_learning_hypers)

    def _update_indices(self):
        self.active_dataset.indices = np.where(self.acquisition_mask)[0]
        self.available_dataset.indices = np.where(~self.acquisition_mask)[0]

    def _update_weights(self, _run):
        if self.weighting_scheme == "none":
            pass
        elif self.weighting_scheme == "refined":
            self._update_refined_weight_scheme(_run)
        elif self.weighting_scheme == "naive":
            self._update_naive(_run)
        else:
            raise NotImplementedError

    def acquire_points_and_update_weights(
        self, scores, active_learning_hypers, _run=None, logging=None
    ):
        probability_masses = scores / torch.sum(scores)
        proposal = active_learning_hypers["proposal"]
        if proposal == "proportional":
            idxs, masses = sample_proportionally(
                probability_masses, active_learning_hypers
            )
        elif proposal == "softmax":
            idxs, masses = sample_softmax(probability_masses, active_learning_hypers)
        else:
            raise NotImplementedError

        # This index is on the set of points in "available_dataset"
        # This maps onto an index in the train_dataset (as opposed to valid)
        train_idxs = get_subset_base_indices(self.available_dataset, idxs)
        # Then that maps onto the index in the original union of train and validation
        true_idxs = get_subset_base_indices(
            self.dataset, train_idxs
        )  # These are the 'canonical' indices to add to the acquired points

        self.dataset.dataset.proposal_mass[true_idxs] = masses
        self.dataset.dataset.sample_order[true_idxs] = int(
            torch.max(self.dataset.dataset.sample_order) + 1
        )
        self.acquisition_mask[train_idxs] = True
        self._update_weights(_run)

        if _run is not None:
            if logging is not None:
                if logging["images"]:
                    # save the mnist image to a jpg
                    acquired_pixels = self.dataset.dataset.data[true_idxs]
                    Image.fromarray(acquired_pixels[0].numpy(), "L").save(
                        "tmp/temp.jpg"
                    )
                    _run.add_artifact(
                        "tmp/temp.jpg", f"{len(self.active_dataset.indices)}.jpg"
                    )
                if logging["classes"]:
                    _run.log_scalar(
                        "acquired_class", f"{self.dataset.dataset.targets[true_idxs]}"
                    )
                    #print(f"Picked: {self.dataset.dataset.targets[true_idxs]}")
                    num_acquired_points = len(self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0])
                    class_distribution = [torch.sum(c==self.dataset.dataset.targets[self.dataset.dataset.sample_order > 0], dtype=torch.float32) / num_acquired_points for c in range(0,10)]
                    print(f"Classes: {class_distribution}")

        self._update_indices()

    def _update_refined_weight_scheme(self, _run):
        """
        This does the work for the method known as R_lure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (N - m + 1) * q
        weight = 1 / weight - 1
        weight = (N - M) * weight
        weight = weight / (N - m)
        weight = weight + 1
        #print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)

    def _update_naive(self, _run):
        """
        This does the work for the method known as R_pure"""
        N = len(self.active_dataset) + len(
            self.available_dataset
        )  # Note that self.dataset.datset includes validation, which should not be in N!
        M = torch.sum(self.dataset.dataset.sample_order > 0)
        active_idxs = self.dataset.dataset.sample_order > 0
        m = self.dataset.dataset.sample_order[active_idxs]
        q = self.dataset.dataset.proposal_mass[active_idxs]

        weight = (1 / q) + M - m
        weight = weight / N
        #print(f"New weight {weight.cpu().numpy()[m.numpy().argsort()]}")
        self.dataset.dataset.weights[active_idxs] = weight.float()
        self.log_weights(_run, weight, m, M)


    def log_weights(self, _run, weight, m, M):
        # Lets log the new weights for the datapoints:
        if _run is not None:
            if "weights" not in _run.info:
                _run.info["weights"] = collections.OrderedDict()
            M = str(M.numpy())
            ordering = m.numpy().argsort()
            _run.info["weights"][M] = weight.numpy()[ordering].tolist()


class RandomFixedLengthSampler(Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    def __init__(self, dataset, target_length):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        assert self.target_length >= len(self.dataset)

        return iter((torch.randperm(self.target_length) % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length


def load_data(
    dataset,
    training_hypers,
    active_learning_hypers,
    train_class_balance,
    test_class_balance,
    training_size_restriction,
    is_active_evaluation=False,
):
    if dataset == "MNIST":
        dataset = WeightedMNIST
    elif dataset == "two_moons":
        dataset = WeightedTwoMoons
    elif dataset == "FashionMNIST":
        dataset = WeightedFashionMNIST
    # When doing active evaluation, the train dataset is actually test data.
    use_train_data = True
    data_noise_proportion = training_hypers["data_noise_proportion"]
    if is_active_evaluation:
        use_train_data = False
        data_noise_proportion = None
    train_dataset = dataset(
        "data",
        class_balance=train_class_balance,
        data_noise_proportion=data_noise_proportion,
        train=use_train_data,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    if training_hypers["num_repetitions"] > 1:
        raise NotImplementedError
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset] * training_hypers["num_repetitions"]
        )

    test_loader = DataLoader(
        dataset(
            "data",
            class_balance=test_class_balance,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=training_hypers["batch_size"] * 8,
        shuffle=True,
        num_workers=training_hypers["num_workers"],
        pin_memory=training_hypers["pin_memory"],
    )

    if training_size_restriction is None:
        train_dataset, validation_dataset = random_split(
            train_dataset,
            [
                len(train_dataset) - training_hypers["validation_set_size"],
                training_hypers["validation_set_size"],
            ],
        )
    else:
        train_dataset, _, validation_dataset = random_split(
            train_dataset,
            [
                training_size_restriction,
                len(train_dataset)
                - training_hypers["validation_set_size"]
                - training_size_restriction,
                training_hypers["validation_set_size"],
            ],
        )

    active_learning_data = ActiveLearningData(train_dataset, active_learning_hypers)

    if training_hypers["padding_epochs"] != "none":
        sampler = RandomFixedLengthSampler(
            active_learning_data.active_dataset, training_hypers["padding_epochs"]
        )
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.active_dataset,
        sampler=sampler,
        shuffle=False,
        batch_size=training_hypers["batch_size"],
        num_workers=training_hypers["num_workers"],
        pin_memory=training_hypers["pin_memory"],
    )

    available_loader = torch.utils.data.DataLoader(
        active_learning_data.available_dataset,
        batch_size=training_hypers["batch_size"],
        shuffle=False,
        num_workers=training_hypers["num_workers"],
        pin_memory=training_hypers["pin_memory"],
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=training_hypers["batch_size"] * 8,
        shuffle=False,
        num_workers=training_hypers["num_workers"],
        pin_memory=training_hypers["pin_memory"],
    )

    return (
        train_loader,
        available_loader,
        active_learning_data,
        validation_loader,
        test_loader,
    )
