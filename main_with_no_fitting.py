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
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import seaborn as sns

from active_learning_utils import (
    get_balanced_samples,
    get_target_classes,
    score_available_data,
    kde_heuristic
)
from dataloader import load_data
from models import ToyBNN, BNN, ConsistentBNN, RadialBNN
from radial_layers.loss import Elbo


ex = Experiment("MNIST")
ex.observers.append(FileStorageObserver("sacred/active-evaluation/mcdo/100"))
# config_path = os.path.join("config", "mnist_test.json")
# ex.add_config(config_path)


@ex.config
def default():
    # See "main.py" for a description of the config
    active_learning_hypers = {
        "num_to_acquire": 1,
        "starting_points": 100,
        "balanced_start": False,
        "scoring_variational_samples": 100,
        "score_strategy": "mutual_information",
        "weighting_scheme": "refined",
        "warm_start": False,
        "mi_plotting": True,
        "proposal": "proportional",
        "temperature": 10000,
    }

    training_hypers = {
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "training_variational_samples": 8,
        "validation_set_size": 792,
        "num_workers": 4,
        "pin_memory": True,
        "early_stopping_epochs": 20,
        "padding_epochs": "none",
        "num_repetitions": 1,
        "test_only": True,
        "weight_decay": 1e-4,
        "model": "mcdo_bnn",
        "channels": 16,
        "checkpoints_frequency": 3,
        "data_noise_proportion": 0.,
    }

    test_class_balance = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1]
    train_class_balance = [a / 4 for a in test_class_balance]
    test_class_balance = [a / 4 for a in test_class_balance]
    # class_balance = [0.05 for i in range(10)]
    # class_balance = None
    training_size_restriction = None
    logging = {"images": False, "classes": True}

    goal_points = 70

    dataset = "MNIST"

    model_save_dir = Path("sacred", "active-evaluation", "models", dataset)
    model_save_dir.mkdir(parents=True, exist_ok=True)


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
    model.train()
    # avg_train_loss = 0
    if model_arch == "radial_bnn":
        loss_object = Elbo(binary=False, regression=False)
        loss_object.set_model(model, train_loader.batch_size)
        loss_object.set_num_batches(len(train_loader))

        def loss_helper(prediction, target):
            nll_loss, kl_loss = loss_object.compute_loss(prediction, target)
            return nll_loss + kl_loss / 10

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


def evaluate(model, eval_loader, training_hypers, active_learning_hypers, n_points=None):
    # We actually only want eval mode on when we're doing acquisition because of how consistent dropout works.
    model_arch = training_hypers["model"]
    n_samples = training_hypers["training_variational_samples"]
    weighting_scheme = active_learning_hypers["weighting_scheme"]
    model.train()

    nll = 0
    weighted_nll = 0
    correct = 0

    n_points_so_far = 0
    if n_points is None:
        n_points = len(eval_loader.dataset)
    with torch.no_grad():
        for data_N_C_H_W, target_N, weight_N in eval_loader:
            if n_points_so_far + data_N_C_H_W.shape[0] > n_points:
                # This batch would be exceeding the number of points we plan to do.
                points_still_allowed = n_points - n_points_so_far
                data_N_C_H_W = data_N_C_H_W[0:points_still_allowed, :]
                target_N = target_N[0:points_still_allowed]
                weight_N = weight_N[0:points_still_allowed]
            n_points_so_far += data_N_C_H_W.shape[0]
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
                weighted_nll = 0.0
            else:
                weighted_nll += torch.sum(weight_N * raw_nll_N)

            # get the index of the max log-probability
            class_prediction = prediction_N.max(1, keepdim=True)[1]
            correct += (
                class_prediction.eq(target_N.view_as(class_prediction)).sum().item()
            )
            if n_points_so_far == n_points:
                # We've done what we need
                break
            elif n_points_so_far > n_points:
                # This should never happen
                raise Exception

    if n_points is not None:
        nll /= n_points
    else:
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
    print(
        f"Beginning training with {len(train_loader.dataset)} training points and {len(validation_loader.dataset)} validation."
    )
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
            model, validation_loader, training_hypers, active_learning_hypers
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


def train_or_load_model(
    model_path,
    training_hypers,
    active_learning_hypers,
    dataset,
    train_class_balance,
    test_class_balance,
    training_size_restriction,
    _run,
):
    model = get_model(training_hypers)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        print(f"Model already found at {model_path}")
    else:
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
        model = train_to_convergence(
            model,
            train_loader,
            validation_loader,
            training_hypers,
            active_learning_hypers,
            _run,
            for_acquisition=True,
        )
        # This is our model, let's save it:
        torch.save(model.state_dict(), model_path)
    return model


@ex.capture
def experiment(
    _run,
    dataset,
    train_class_balance,
    test_class_balance,
    training_size_restriction,
    goal_points,
    model_save_dir,
    training_hypers=None,
    active_learning_hypers=None,
    logging=None,
):
    """
    This is the experiment for Figure 2.
    We train a fixed model and then use an active learning loop to evaluate the risk of that model over a small number of points.
    This set-up is what is referred to in Kossen et al. 2021 as "Active Testing"
    """

    # First we just train a model on some points.
    model_path = model_save_dir / (
        training_hypers["model"] + f"{active_learning_hypers['starting_points']}.pth"
    )
    model = train_or_load_model(
        model_path,
        training_hypers,
        active_learning_hypers,
        dataset,
        train_class_balance,
        test_class_balance,
        training_size_restriction,
        _run,
    )

    # Now, we use the active loop to evaluate it.
    # First, we remake the dataloaders. The loaders are designed for doing active *learning*
    # So we will test using the trainloader, but tweaked to call the test dataset to avoid overlap
    testing_hypers = copy.deepcopy(training_hypers)
    active_testing_hypers = copy.deepcopy(active_learning_hypers)
    active_testing_hypers["starting_points"] = 0

    print("Loading test data...", end="")
    (
        active_test_loader,
        available_loader,
        active_learning_data,
        validation_loader,
        full_test_loader,
    ) = load_data(
        dataset,
        testing_hypers,
        active_testing_hypers,
        test_class_balance,
        test_class_balance,
        training_size_restriction,
        is_active_evaluation=True,
    ) # Duplicate test_class_balance for both, since the train data is also test
    print("done.")

    active_test_size = len(active_test_loader.dataset)
    empirical_risk = None
    while active_test_size <= goal_points:
        pool_size = len(available_loader.dataset)

        # print(f"Computing scores for {pool_size} points...", end="")
        # scores = score_available_data(
        #     model,
        #     available_loader,
        #     training_hypers["model"],
        #     active_learning_hypers["scoring_variational_samples"],
        #     score_strategy=active_learning_hypers["score_strategy"],
        #     plotting=False,
        #     step=len(active_test_loader.dataset),
        #     _run=_run,
        # )

        # We use the KDE heuristic for the 1-d data
        scores = kde_heuristic(available_loader, active_test_loader)
        # print("done.")

        # print("Acquiring new point(s)...", end="")
        active_learning_data.acquire_points_and_update_weights(
            scores, active_learning_hypers, logging=logging, _run=_run,
        )
        # print("done.")
        active_test_size = len(active_test_loader.dataset)

        # Now we want to know:
        # r - population risk, not knowable. Damn!
        # Question to the meaning of life, not knowable. Damn!
        # \hat{R} - empirical risk on the full_test_loader. (only once)
        # empirical risk on randomly acquired data of the same size. (each time)
        # \tilde{R} - unweighted risk on active_test_loader
        # \tilde{R}_\textup{PURE} - PURE risk estimator on active_test_loader
        # \tilde{R}_\textup{LURE} - LURE risk estimator on active_test_loader (note this might sometimes be written as "sure" in the code as this was the name of the method until reviews)

        if empirical_risk is None:
            empirical_risk, _, empirical_accuracy = evaluate(model, full_test_loader, testing_hypers, active_testing_hypers)
        subsample_empirical_risk, _, _ = evaluate(model, full_test_loader, testing_hypers, active_testing_hypers, n_points=active_test_size)
        unweighted_risk, _, _ = evaluate(model, active_test_loader, testing_hypers, active_testing_hypers)
        # making sure we have PURE - using old naming scheme
        active_learning_data.weighting_scheme = "naive"
        # This was not designed to be called outside the acquisition loop, but doing it this way
        # lets us compute both estimators on the same run
        active_learning_data._update_weights(_run)
        _, pure_risk, _ = evaluate(model, active_test_loader, testing_hypers, active_testing_hypers)
        # making sure we have LURE - using old naming scheme
        active_learning_data.weighting_scheme = "refined"
        active_learning_data._update_weights(_run)
        _, sure_risk, _ = evaluate(model, active_test_loader, testing_hypers, active_testing_hypers)

        _run.log_scalar("empirical_risk", empirical_risk, step=active_test_size)
        _run.log_scalar("subsample_empirical_risk", subsample_empirical_risk, step=active_test_size)
        _run.log_scalar("unweighted_risk", unweighted_risk, step=active_test_size)
        _run.log_scalar("pure_risk", pure_risk, step=active_test_size)
        _run.log_scalar("sure_risk", sure_risk, step=active_test_size)
        _run.log_scalar("empirical_accuracy", empirical_accuracy, step=active_test_size)

        print(f"With {active_test_size} points:")
        print(f"Empirical Risk: {empirical_risk}")
        print(f"SubEmp Risk   : {subsample_empirical_risk}")
        print(f"Unweighted    : {unweighted_risk}")
        print(f"PURE          : {pure_risk}")
        print(f"SURE          : {sure_risk}")


@ex.automain
def main():
    experiment()
