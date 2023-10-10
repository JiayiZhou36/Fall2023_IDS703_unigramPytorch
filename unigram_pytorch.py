"""Pytorch."""
from typing import List, Optional
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))
    # dim =1 : sum up row wise
    rows = torch.sum(x, dim=1)
    # the number a character appears divide by the total time
    opt_prob = rows / (torch.sum(rows))
    # the count of time appears * log probability
    opt_loss = (torch.sum(x, 1, keepdim=True).T @ torch.log(opt_prob)) * -1

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 1000
    learning_rate = 0.1

    # train model
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        loss_list.append(loss)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # display results
    # plot loss as a function and minimum possible loss
    loss_list = torch.cat(loss_list, dim=0).detach().numpy()
    opt_loss_list = [k.item() for k in opt_loss]
    plt.figure(1)
    plt.plot(loss_list, label="Training Loss")
    plt.axhline(
        opt_loss_list, color="r", linestyle="--", label="Optimal Loss"
    )  # Add the optimal loss line
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("The Loss as a Function of Time")
    plt.legend()  # Show the legend

    ##Figure 2--plot final token probabilities
    parameter_list = normalize(torch.sigmoid(next(model.parameters())))
    parameter_list = [p.item() for p in parameter_list]
    vocabulary_list = [chr(i + ord("a")) for i in range(26)] + [
        " ",
        "N",
    ]
    plt.figure(2)
    plt.bar(vocabulary_list, parameter_list, label="Token Probability")
    plt.xlabel("Vocabulary")
    plt.ylabel("Probability")
    plt.title("The Final Token Probabilities")
    plt.legend()  # Show the legend

    ## Figure 3--plot optimal probabilities
    plt.figure(3)
    opt_prob_list = [l.item() for l in opt_prob]
    plt.bar(vocabulary_list, opt_prob_list, label="Optimal Probability")
    plt.xlabel("Vocabulary")
    plt.ylabel("Probability")
    plt.title("The Optimal Probabilities")
    plt.legend()  # Show the legend
    plt.show()


if __name__ == "__main__":
    gradient_descent_example()
