import os
import math
import copy
import json

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, random_split, DataLoader, Sampler
from torchvision import datasets, transforms

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import seaborn as sns

from active_learning_utils import (
    get_balanced_samples,
    get_target_classes,
    score_available_data,
)
from dataloader import load_data
from models import ToyBNN, BNN, ConsistentBNN, RadialBNN
from radial_layers.loss import Elbo


ex = Experiment("FashionMNIST")
ex.observers.append(FileStorageObserver("sacred/rebuttal-fig3bc-radial-fashion/"))
# config_path = os.path.join("config", "mnist_test.json")
# ex.add_config(config_path)


@ex.config
def default():
    # valid score strategies are:
    # mutual_information
    # entropy
    # random_acquisition

    # valid weighting_scheme are
    # none = just an unweighted MC estimator
    # naive = Referred to in the paper as R_{pure}
    # refined = Referred to in the paper as R_{lure}

    active_learning_hypers = {
        "num_to_acquire": 1,  # Number of points acquired each time we do acquisition
        "starting_points": 10,  # Number of points we start with (randomly selected)
        "balanced_start": False,  # Ensure there's an equal number from each class at initialization
        "scoring_variational_samples": 100,  # Number of samples from posterior to take when we acquire
        "score_strategy": "mutual_information",  # May be "mutual_information", "entropy", or "random_acquisition"
        "weighting_scheme": "refined",  # May be "none", "naive", or "refined"
        "warm_start": False,  # Use the last trained model to initialize the new training loop
        "mi_plotting": True,  # Plot the MI's of the pool for diagnostic purposes
        "proposal": "softmax",  # Can be "softmax" or "proportional" for how we sample w.r.t. the scores
        "temperature": 15000  # Temperature of the softmax distribution if being used
    }

    training_hypers = {
        "max_epochs": 100,  # Hard cut-off number of epochs to train each time
        "learning_rate": 1e-4,
        "batch_size": 64,
        "training_variational_samples": 8,
        "validation_set_size": 1000,
        "num_workers": 4,
        "pin_memory": True,
        "early_stopping_epochs": 20,  # The number of epochs to be 'patient' for
        "padding_epochs": 1000,
        "num_repetitions": 1,  # I don't think this should ever be used but am nervous to take it out in case it breaks something
        "test_only": True,
        "weight_decay": 1e-4,
        "model": "mcdo_bnn",  # Can be "toy_bnn", "mcdo_bnn", "radial_bnn", "consistent_mcdo"
        "channels": 16,
        "checkpoints_frequency": 3,
        "data_noise_proportion": 0.1,
    }

    test_class_balance = [1., 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.01, 0.01]
    train_class_balance = [a/4 for a in test_class_balance]
    # class_balance = [0.05 for i in range(10)]
    # class_balance = None
    training_size_restriction = None
    logging = {"images": False, "classes": True}

    goal_points = 70

    dataset = "MNIST"  # May be "MNIST" "FashionMNIST" or "two_moons"


def train(
    model,
    train_loader,
    optimizer,
    weighting_scheme,
    training_variational_samples,
    model_arch,
    padding_epochs,
    _run=None,
    for_acquisition=False,
    test_only=False,
):
    """
    Helper function to execute a single epoch of training.
    """
    model.train()
    # avg_train_loss = 0
    if model_arch == "radial_bnn":
        loss_object = Elbo(binary=False, regression=False)
        loss_object.set_model(model, train_loader.batch_size)
        loss_object.set_num_batches(len(train_loader))

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss_object.compute_loss(prediction, target)
            return nll_loss + kl_loss /10

        raw_loss = loss_helper
    else:
        raw_loss = torch.nn.NLLLoss()

    # losses = []
    for batch_idx, (data_N_C_H_W, target, weight) in enumerate(train_loader):
        data_N_C_H_W = data_N_C_H_W.cuda()
        target = target.cuda()
        weight = weight.cuda()

        optimizer.zero_grad()

        if model_arch == "consistent_mcdo":
            prediction = torch.logsumexp(
                model(data_N_C_H_W, training_variational_samples), dim=1
            ) - math.log(training_variational_samples)
        elif model_arch == "radial_bnn":
            assert len(data_N_C_H_W.shape) == 4
            data_N_V_C_H_W = torch.unsqueeze(data_N_C_H_W, 1)
            data_N_V_C_H_W = data_N_V_C_H_W.expand(
                -1, training_variational_samples, -1, -1, -1
            )
            prediction = model(data_N_V_C_H_W)
        else:
            prediction = model(data_N_C_H_W)  # Always uses 1 when not doing consistent.

        if weighting_scheme == "none" or (for_acquisition and test_only):
            loss = raw_loss(prediction, target).mean(dim=0)
        else:
            loss = (weight * raw_loss(prediction, target)).mean(0)

        loss.backward()
        # avg_train_loss = (avg_train_loss * batch_idx + loss.item()) / (batch_idx + 1)
        _run.log_scalar("training_loss", loss.item())
        optimizer.step()

        # losses.append(loss.item())
        # print(f'Epoch: {epoch}:')
    # print(f'Train Set: Average Loss: {avg_train_loss:.6f}')


def evaluate(model, eval_loader, training_hypers, active_learning_hypers):
    """We actually only want eval mode on when we're doing acquisition because of how consistent dropout works.
    """
    model_arch = training_hypers["model"]
    n_samples = training_hypers["training_variational_samples"]
    weighting_scheme = active_learning_hypers["weighting_scheme"]
    model.train()

    nll = 0
    weighted_nll = 0
    correct = 0

    with torch.no_grad():
        for data_N_C_H_W, target_N, weight_N in eval_loader:
            data_N_C_H_W = data_N_C_H_W.cuda()
            target_N = target_N.cuda()
            weight_N = weight_N.cuda()

            if model_arch == "consistent_mcdo":
                prediction_N = torch.logsumexp(
                    model(data_N_C_H_W, n_samples), dim=1
                ) - math.log(n_samples)
            elif model_arch == "radial_bnn":
                data_N_V_C_H_W = torch.unsqueeze(data_N_C_H_W, 1)
                data_N_V_C_H_W = data_N_V_C_H_W.expand((-1, n_samples, -1, -1, -1))
                prediction_N = torch.logsumexp(model(data_N_V_C_H_W), dim=1) - math.log(
                    n_samples
                )
            else:
                samples_V_N = torch.stack(
                    [model(data_N_C_H_W) for _ in range(n_samples)]
                )
                prediction_N = torch.logsumexp(samples_V_N, dim=0) - math.log(n_samples)

            raw_nll_N = F.nll_loss(prediction_N, target_N, reduction="none")
            nll += torch.sum(raw_nll_N)
            if weighting_scheme == "none":
                weighted_nll = 0.
            else:
                weighted_nll += torch.sum(weight_N * raw_nll_N)

            # get the index of the max log-probability
            class_prediction = prediction_N.max(1, keepdim=True)[1]
            correct += (
                class_prediction.eq(target_N.view_as(class_prediction)).sum().item()
            )

    nll /= len(eval_loader.dataset)
    if weighting_scheme == "none":
        pass
    else:
        weighted_nll /= len(eval_loader.dataset)
        weighted_nll = weighted_nll.item()
    percentage_correct = 100.0 * correct / len(eval_loader.dataset)

    return nll.item(), weighted_nll, percentage_correct


def train_to_convergence(
    model,
    train_loader,
    validation_loader,
    training_hypers,
    active_learning_hypers,
    _run,
    for_acquisition,
):
    """
    Helper function to train multiple epochs until the set limit is reached or we lose patience.
    """
    print(f"Beginning training with {len(train_loader.dataset)} training points and {len(validation_loader.dataset)} validation.")
    best = np.inf
    best_model = model
    patience = 0
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_hypers["learning_rate"],
        weight_decay=training_hypers["weight_decay"],
    )
    print(f"Digits used: {len(train_loader.dataset)}")
    for epoch in range(training_hypers["max_epochs"]):
        train(
            model,
            train_loader,
            optimizer,
            active_learning_hypers["weighting_scheme"],
            training_hypers["training_variational_samples"],
            training_hypers["model"],
            training_hypers["padding_epochs"],
            _run=_run,
            for_acquisition=for_acquisition,
            test_only=training_hypers["test_only"],
        )
        valid_nll, _, valid_accuracy = evaluate(
            model,
            validation_loader,
            training_hypers,
            active_learning_hypers
        )
        # _run.log_scalar("evaluation_loss", valid_loss)
        # _run.log_scalar("evaluation_accuracy", valid_accuracy)
        print(
            f"Epoch {epoch:0>3d} eval: Val nll: {valid_nll:.4f}, Val Accuracy: {valid_accuracy}"
        )

        if valid_nll < best:
            best = valid_nll
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1

        if patience >= training_hypers["early_stopping_epochs"]:
            print(f"Patience reached - stopping training. Best was {best}")
            break
    print("Completed training", end="")
    if for_acquisition and training_hypers["test_only"]:
        print(" for acquisition.")
    else:
        print(".")
    return best_model


def get_model(training_hypers):
    """
    Snag the requested model given the config file
    """
    model_arch = training_hypers["model"]
    channels = training_hypers["channels"]
    if model_arch == "toy_bnn":
        model = ToyBNN(0.5)
    elif model_arch == "mcdo_bnn":
        model = BNN(0.5, channels)
    elif model_arch == "consistent_mcdo":
        model = ConsistentBNN()
    elif model_arch == "radial_bnn":
        model = RadialBNN(channels)
    else:
        print(model_arch)
        raise NotImplementedError
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA NOT AVAILABLE! CONTINUING ON CPU!")
    return model


@ex.capture
def experiment(
    _run,
    dataset,
    train_class_balance,
    test_class_balance,
    training_size_restriction,
    goal_points,
    training_hypers=None,
    active_learning_hypers=None,
    logging=None,
):
    """
    Train models and acquire points given the score function that has been chosen. Every few models we log the performance
    of a model trained *unweighted* and *weighted* for comparison.
    """
    checkpoints_frequency = training_hypers["checkpoints_frequency"]
    print("Loading data...", end="")
    (
        train_loader,
        available_loader,
        active_learning_data,
        validation_loader,
        test_loader,
    ) = load_data(
        dataset,
        training_hypers,
        active_learning_hypers,
        train_class_balance,
        test_class_balance,
        training_size_restriction,
    )
    print("done.")

    model = None
    train_size = len(train_loader.dataset)
    while train_size <= goal_points:
        train_size = len(train_loader.dataset)
        pool_size = len(available_loader.dataset)
        if model is None or not active_learning_hypers["warm_start"]:
            model = get_model(training_hypers)
        model = train_to_convergence(
            model,
            train_loader,
            validation_loader,
            training_hypers,
            active_learning_hypers,
            _run,
            for_acquisition=True,
        )

        (
            acquisition_test_nll,
            acquisition_test_weighted_nll,
            acquisition_test_accuracy,
        ) = evaluate(model, test_loader, training_hypers, active_learning_hypers)
        _run.log_scalar("acquisition_test_nll", acquisition_test_nll, step=train_size)
        _run.log_scalar(
            "acquisition_test_weighted_nll",
            acquisition_test_weighted_nll,
            step=train_size,
        )
        _run.log_scalar(
            "acquisition_test_accuracy", acquisition_test_accuracy, step=train_size,
        )

        print(
            f"Test set: Average loss: {acquisition_test_nll:.4f}, Accuracy: {acquisition_test_accuracy}"
        )
        plot_scores = False
        if train_size % checkpoints_frequency == 0:
            # In this case, we train with the corrected estimator
            clean_model = get_model(training_hypers)
            clean_model = train_to_convergence(
                clean_model,
                train_loader,
                validation_loader,
                training_hypers,
                active_learning_hypers,
                _run,
                for_acquisition=False,
            )
            clean_test_nll, _, clean_test_accuracy = evaluate(
                clean_model, test_loader, training_hypers, active_learning_hypers
            )
            _run.log_scalar("test_nll", clean_test_nll, step=train_size)
            _run.log_scalar("test_accuracy", clean_test_accuracy, step=train_size)
            print(
                f"Test with {train_size} points using weighted training: Average loss: {clean_test_nll:.4f}, Accuracy: {clean_test_accuracy}"
            )

            # We want to see if L_weighted(train) is a good predictor of L_raw(train + pool)
            clean_train_nll, clean_train_weighted_nll, _ = evaluate(
                clean_model, train_loader, training_hypers, active_learning_hypers
            )
            clean_pool_nll, _, _ = evaluate(
                clean_model, available_loader, training_hypers, active_learning_hypers
            )
            clean_joint_nll = (
                train_size * clean_train_nll + pool_size * clean_pool_nll
            ) / (train_size + pool_size)
            _run.log_scalar("weighted_train_joint_nll", clean_joint_nll, step=train_size)
            _run.log_scalar("weighted_train_unweighted_nll", clean_train_nll, step=train_size)
            _run.log_scalar(
                "weighted_train_weighted_nll", clean_train_weighted_nll, step=train_size
            )

            # And also for the unweighted model
            if training_hypers["test_only"]:
                train_nll, train_weighted_nll, _ = evaluate(
                    model, train_loader, training_hypers, active_learning_hypers
                )
                pool_nll, _, _ = evaluate(
                    model, available_loader, training_hypers, active_learning_hypers
                )
                joint_nll = (
                                      train_size * train_nll + pool_size * pool_nll
                                  ) / (train_size + pool_size)
                _run.log_scalar("train_joint_nll", joint_nll, step=train_size)
                _run.log_scalar("train_unweighted_nll", train_nll, step=train_size)
                _run.log_scalar(
                    "train_weighted_nll", train_weighted_nll, step=train_size
                )

            if active_learning_hypers["mi_plotting"]:
                plot_scores = True

        print(f"Computing scores for {len(available_loader.dataset)} points...", end="")
        scores = score_available_data(
            model,
            available_loader,
            training_hypers["model"],
            active_learning_hypers["scoring_variational_samples"],
            score_strategy=active_learning_hypers["score_strategy"],
            plotting=plot_scores,
            step=len(train_loader.dataset),
            _run=_run,
        )
        print("done.")

        print("Acquiring new point(s)...", end="")
        active_learning_data.acquire_points_and_update_weights(
            scores,
            active_learning_hypers,
            logging=logging,
            _run=_run,
        )
        print("done.")

    return clean_model


@ex.automain
def main():
    experiment()
