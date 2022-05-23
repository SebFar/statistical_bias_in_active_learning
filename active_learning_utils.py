import itertools
import math
import torch
import numpy as np


def get_balanced_samples(target_classes, n_per_digit):
    """
    Returns a sample from the data with the same number of samples from each targeted class, otherwise uniform.
    
    Not currently used.
    TODO: Write proper comments if it becomes used.
    
    :param target_classes: 
    :param n_per_digit: 
    :return: 
    """
    sample_indices = {}

    for digit in range(10):
        digit_set = target_classes == digit
        sample_indices[digit] = np.random.choice(
            np.where(digit_set)[0], size=(n_per_digit,), replace=False
        )

    initial_samples = list(itertools.chain.from_iterable(sample_indices.values()))

    return initial_samples


def get_target_classes(dataset):
    return np.array(dataset, dtype=object)[:, 1].astype(int)


def get_top_n(scores, n):
    top_n = np.argpartition(scores, -n)[-n:]
    #     print(f'Top n scores: {scores[top_n]}')
    return top_n


def toy_score(model, available_loader, consistent_dropout, n_samples):
    scores = []
    with torch.no_grad():
        for data, _, _ in available_loader:
            scores.append(torch.abs(data[:, 0]))
        scores = torch.cat(scores)
    return scores.cpu()


def mutual_information(
    model,
    available_loader,
    model_arch,
    n_samples,
    plotting=False,
    step=0,
    _run=None,
):
    # For consistent dropout this is critical to get the right behaviour
    model.eval()
    multi_scores_S_N = []

    if not plotting:
        num_score_samples = 1
    else:
        num_score_samples = 5

    for score_counter in range(num_score_samples):
        scores_N = []
        with torch.no_grad():
            for data, _, _ in available_loader:
                data = data.cuda()

                if model_arch == "consistent_mcdo":
                    samples_V_N_K = model(data, n_samples).permute(1, 0, 2)
                elif model_arch == "radial_bnn":
                    data = torch.unsqueeze(data, 1)
                    data = data.expand((-1, n_samples, -1, -1, -1))
                    samples_V_N_K = model(data).permute(1, 0, 2)
                else:
                    samples_V_N_K = torch.stack([model(data) for _ in range(n_samples)])
                average_entropy_N = -torch.sum(
                    samples_V_N_K.exp() * samples_V_N_K, dim=2
                ).mean(0)

                mean_samples_N_K = torch.logsumexp(samples_V_N_K, dim=0) - math.log(
                    n_samples
                )
                entropy_average_N = -torch.sum(
                    mean_samples_N_K.exp() * mean_samples_N_K, dim=1
                )

                score_N = entropy_average_N - average_entropy_N
                scores_N.append(score_N)

        scores_N = torch.cat(scores_N)
        multi_scores_S_N.append(scores_N)
    if plotting:
        # now we want to see both how flat the mutual_information is and how much variation there is.
        multi_scores_S_N = torch.stack(multi_scores_S_N, dim=0)
        mean_scores_N = multi_scores_S_N.mean(dim=0)
        sorting_idxs = torch.sort(mean_scores_N)[1]
        mean_scores_N = mean_scores_N[sorting_idxs].cpu()
        std_scores_N = multi_scores_S_N.std(dim=0)[sorting_idxs].cpu()
        # import matplotlib.pyplot as plt
        #
        # N = np.arange(len(mean_scores_N))
        # plt.plot(N, mean_scores_N, label=f"{step}")
        # plt.fill_between(
        #     N, mean_scores_N - std_scores_N, mean_scores_N + std_scores_N, alpha=0.4
        # )
        # plt.xlabel("Datapoints")
        # plt.ylabel("MI")
        # plt.savefig(f"tmp/{step}.png", bbox_inches="tight", dpi=300)
        if _run is not None:
            if "mi" not in _run.info:
                _run.info["mi"] = {}
            _run.info["mi"][str(step)] = [
                mean_scores_N.numpy().tolist(),
                std_scores_N.numpy().tolist(),
            ]

    # Occassionally a noisy prediciton on a very small MI might lead to negative score
    # We knock these up to 0
    scores_N[scores_N < 0] = 0.
    return scores_N.cpu()


def entropy_score(model, available_loader, n_samples):
    """
    This is the expected loss on that data point assuming that the data distribution is the current proposal
    - \Sum_{h \in C} 1/K \Sum_{i} p(y=h|X)q(w_i) log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
    :param model:
    :param available_loader:
    :param n_samples:
    :return:
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for data, _, _ in available_loader:
            data = data.cuda()
            # Shape of samples is [variational_samples, batch_size, n_classes]
            samples = torch.stack([model(data) for _ in range(n_samples)])
            # mean_samples is log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            # Shape is [batch_size, n_classes]
            mean_samples = torch.logsumexp(samples, dim=0) - math.log(n_samples)
            # score is \Sum_{h \in C} 1/K \Sum_{i} p(y=h|X)q(w_i) log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            score = -torch.sum(mean_samples.exp() * mean_samples, dim=1)
            scores.append(score)
        scores = torch.cat(scores)
    return scores


def grad_entropy_score(model, available_loader, n_samples):
    """
    This is the expected gradient of the loss on that data point assuming that the data distribution is the current proposal
    \E_{p(y=c|x) \sim D} \grad - \log \E_{\theta} p(y=c|x, \theta)
    - \Sum_{c \in C} 1/K \Sum_{i} p(y=h|X)q(w_i) \grad_w_i log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
    :param model:
    :param available_loader:
    :param n_samples:
    :return:
    """
    model.eval()
    scores = []
    for data, _, _ in available_loader:
        data = data.cuda()
        samples = torch.stack([model(data) for _ in range(n_samples)])
        # mean_samples is log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
        mean_samples = torch.logsumexp(samples, dim=0) - math.log(n_samples)
        gradient = torch.autograd.grad(mean_samples, data)
        score = -torch.sum(mean_samples.exp() * gradient, dim=1)
        scores.append(score)
    scores = torch.cat(scores)
    return scores


def uniform_loss_score(model, available_loader, n_samples):
    """
    This is the expected loss on that data point assuming that the data distribution is uniform
    - \Sum_{h \in C} 1/num_classes log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
    :param model:
    :param available_loader:
    :param n_samples:
    :return:
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for data, _, _ in available_loader:
            data = data.cuda()
            samples = torch.stack([model(data) for _ in range(n_samples)])
            # mean_samples is log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            mean_samples = torch.logsumexp(samples, dim=0) - math.log(n_samples)
            # score is \Sum_{h \in C} 1/K \Sum_{i} p(y=h|X)q(w_i) log 1/K \Sum_{i} p(y=h|X, w_i)q(w_i)
            score = -torch.sum(mean_samples / 10, dim=1)
            scores.append(score)
        scores = torch.cat(scores)
    return scores

def kde_heuristic(available_loader, train_loader):
    # We use a mean-squared distance for scoring
    # The scaling for this is pretty bad, but we don't use much data.
    scores = []
    for (available_data_N_C_H_W, available_target_N, availabe_weight_N) in available_loader:
        if len(train_loader.dataset) > 0:
            available_scores_N = torch.zeros_like(available_target_N, dtype=torch.float32)
            for (data_n_C_H_W, target_n, weight_n) in train_loader:
                data_N_n_C_H_W = torch.unsqueeze(data_n_C_H_W, dim=0).expand((len(availabe_weight_N), -1, -1, -1, -1))
                available_data_N_n_C_H_W = torch.unsqueeze(available_data_N_C_H_W, dim=1).expand((-1, len(target_n), -1, -1, -1))
                available_scores_N += torch.sum((available_data_N_n_C_H_W - data_N_n_C_H_W) ** 2, dim=(1,2,3,4))
            scores.append(torch.sqrt(available_scores_N))
        else:
            scores.append(torch.ones_like(available_target_N, dtype=torch.float32))
    scores = torch.cat(scores)

    return scores

def score_available_data(
    model,
    available_loader,
    model_arch,
    n_samples,
    score_strategy,
    plotting=False,
    step=0,
    _run=None,
):
    """
    Computes MI score for each datapoint in the dataset.
    
    Higher score is more uncertainty.
    TODO: Check shape of return.
    
    :param model: Model being evaluated. 
    :param available_loader: Dataloader containing scorable points (the unlabelled pool)
    :param n_samples: Number of samples from the model to use in estimating the score.
    :return: np_array with shape [n_points]
    """
    if score_strategy == "mutual_information":
        scores = mutual_information(
            model,
            available_loader,
            model_arch,
            n_samples,
            plotting=plotting,
            step=step,
            _run=_run,
        )
    elif score_strategy == "entropy":
        scores = entropy_score(model, available_loader, n_samples)
    elif score_strategy == "uniform_loss":
        scores = uniform_loss_score(model, available_loader, n_samples)
    elif score_strategy == "entropy_gradient":
        scores = grad_entropy_score(model, available_loader, n_samples)
    elif score_strategy == "random_acquisition":
        scores = torch.ones(len(available_loader))
    elif score_strategy == "toy_score":
        scores = toy_score(model, available_loader, consistent_dropout, n_samples)

    return scores
