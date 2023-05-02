import torch
from torch import nn
import numpy as np


class Projector(nn.Module):
    def __init__(
        self,
        input_embedding_dim: int = 300,
        final_embedding_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_embedding_dim, 512)
        self.fn1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, final_embedding_dim)
        self.fn2 = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(final_embedding_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fn2(x)
        x = self.layer_norm(x)
        return x