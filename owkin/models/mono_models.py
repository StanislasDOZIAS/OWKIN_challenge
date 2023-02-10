from torch import nn
from pathlib import Path

from owkin.models.base_model import BaseModel


class MonoModel(BaseModel):
    """
    A Mono Model takes as entry the features of one tile (or equivalent shape).
    """

    def __init__(self):
        super().__init__()
        self.is_aggregator = False
        self.best_path: Path = None


class MLP(MonoModel):
    """
    Multilayer Perceptron.
    The input dim can change if we add some additional features.
    """

    def __init__(self, num_layers=1, inside_dim=0, input_dim=2048):
        super().__init__()
        self.type = "MLP"
        self.inside_dim = inside_dim
        self.num_layers = num_layers

        self.config = {
            "num_layers": num_layers,
            "inside_dim": inside_dim,
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
        return self.layers(x)
