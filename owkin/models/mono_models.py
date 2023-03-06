from torch import nn
import torch
from pathlib import Path

from owkin.models.base_model import BaseModel
from sklearn.svm import SVC

import numpy as np


class MonoModel(BaseModel):
    """
    A Mono Model takes as entry the features of one tile (or equivalent shape).
    The forward method must return Tensors of shape [b,f] or [b]
    (with b the batch size and f the number of features considered)
    """

    def __init__(self):
        super().__init__()
        self.meta_type = "MonoModel"
        self.best_path: Path = None
        self.type = self._get_name()


class MLP(MonoModel):
    """
    Multilayer Perceptron.
    The input dim can change if we add some additional features.
    """

    def __init__(self, num_layers=1, inside_dim=0, input_dim=2048):
        super().__init__()
        self.inside_dim = inside_dim
        self.num_layers = num_layers

        self.config = {
            "nl": num_layers,
            "id": inside_dim,
        }

        self.compute_name()

        if num_layers == 1:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid(),
            )
        else:
            list_layers = [
                nn.Linear(input_dim, inside_dim),
                nn.Dropout(p=0.2),
                nn.ReLU(),
            ]
            for _ in range(num_layers - 2):
                list_layers.append(nn.Linear(inside_dim, inside_dim))
                list_layers.append(nn.Dropout(p=0.2))
                list_layers.append(nn.ReLU())
            list_layers.append(nn.Linear(inside_dim, 1))
            list_layers.append(nn.Sigmoid())
            self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x).squeeze()


class SVM(MonoModel):
    """
    Support Vector Model
    """

    def __init__(self, C=1, kernel="rbf", degree=3):
        super().__init__()
        self.svm = SVC(C=C, kernel=kernel, degree=degree, probability=True)

        self.config = {"C": C, "kernel": kernel, "degree": degree}

    def forward(self, x):
        """Forward pass"""
        if len(x.shape) == 3:
            to_return = np.zeros(x.shape[:-1])
            for i, slide in enumerate(x):
                to_return[i] = self.svm.predict_proba(slide)[:, 1]
        else:
            to_return = self.svm.predict_proba(x)[:, 1]
        return torch.Tensor(to_return)
