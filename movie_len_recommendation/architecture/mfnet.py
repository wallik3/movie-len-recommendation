import pandas as pd
import datetime
from scipy.sparse import csr_matrix
import numpy as np

from sklearn.metrics import recall_score
from typing import List, Tuple
from tqdm import tqdm
import itertools

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F

class MFNet(nn.Module):
    """
    A PyTorch embeddings for matrix factorization.

    Args:
        num_users (int): The number of users in the dataset.
        num_movies (int): The number of movies in the dataset.
        embedding_size (int): The size of the embedding vectors.
    """

    def __init__(self, num_users: int, num_items: int, embedding_size: int):
        """
        Create the user and movie embedding layers.
        """
        super(MFNet, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.item_embeddings = nn.Embedding(num_items, embedding_size)

        # Since the standard of each movie
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute the predicted rating for a given user and movie.

        Args:
            user_idx (torch.Tensor): The index of the user.
            movie_idx (torch.Tensor): The index of the movie.

        Returns:
            torch.Tensor: The predicted rating.
        """
        user_emb = self.user_embeddings(user_idx)  # type: torch.Tensor
        movie_emb = self.item_embeddings(item_idx)  # type: torch.Tensor

        user_bias = self.user_biases(user_idx)
        movie_bias = self.item_biases(item_idx)

        rating = torch.sum(user_emb * movie_emb, dim=1)
        rating = rating + user_bias.squeeze() + movie_bias.squeeze()

        return rating

    def __repr__(self):
        return f"MFNet({self.num_users=}, {self.num_items=}, {self.embedding_size=})"

class MFNetSigmoidRange(nn.Module):
    """
    MFNet but restrict the output range using sigmoud activation function
    """
    def __init__(self, num_users, num_items, embedding_size, output_range:Tuple[float,float]=(0.8,5.2)):
        """
        We recommend the min and max of output range to add with some residual
        since the steepest slope of the min and max is exactly zero, hence it is impossible to learn to predict the exact min and max
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        self.output_range = output_range
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.item_embeddings = nn.Embedding(num_items, embedding_size)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_idx)  # type: torch.Tensor
        item_emb = self.item_embeddings(movie_idx)  # type: torch.Tensor

        user_bias = self.user_biases(user_idx)
        item_bias = self.item_biases(movie_idx)

        rating = torch.sum(user_emb * item_emb, dim=1)
        rating = rating + user_bias.squeeze() + item_bias.squeeze()

        # Post-process to fit with output constraint
        rating = torch.sigmoid(rating)
        rating = rating * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        return rating

    def __repr__(self):
        return f"MFNetSigmoidRange({self.num_users=}, {self.num_items=}, {self.embedding_size=})"