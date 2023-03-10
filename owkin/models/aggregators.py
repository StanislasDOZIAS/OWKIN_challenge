import torch
import copy

from pathlib import Path

from owkin.models.base_model import BaseModel


## TODO: aggregator = instance space method


class BaseAggregator(BaseModel):
    """
    An Aggregator Model takes a Mono Model and apply it to each of the 1000 tiles of a slide.
    Then it aggregates theses results to give a single output (e.g. the Maximum).

    The Mono Model must have:
        - a name (str)
        - a type (str)
        - a config (dict)
    """

    def __init__(self, mono_model):
        super().__init__()
        self.meta_type = "Aggregator"
        self.mono_model = copy.deepcopy(mono_model)

        self.type = f"{self._get_name()}/{self.mono_model.type}"
        self.config = mono_model.config

        self.best_path: Path = None

    def forward(self, x):
        """Forward pass on the 1000 features"""
        raise NotImplementedError


class MaxAggregator(BaseAggregator):
    """
    An aggregator model which outputs the maximum of the 1000 predictions.
    """

    def __init__(self, mono_model):
        super().__init__(mono_model)
        self.compute_name()

    def forward(self, x):
        mono_model_predictions = self.mono_model(x)
        max_prediction, _ = mono_model_predictions.max(axis=-1)
        return max_prediction


class SmoothMaxAggregator(BaseAggregator):
    """
    An aggregator model which outputs the smooth_maximum of the 1000 predictions.
    The idea comes from here: https://arxiv.org/abs/1610.02501, https://arxiv.org/abs/1609.07257
    """

    def __init__(self, mono_model):
        super().__init__(mono_model)
        self.compute_name()

    def forward(self, x):
        mono_model_predictions = self.mono_model(x)
        smooth_max_prediction = (
            torch.log(torch.exp(mono_model_predictions).sum(axis=-1))
            / mono_model_predictions.shape[-1]
        )
        return smooth_max_prediction


class MeanAggregator(BaseAggregator):
    """
    An aggregator model which outputs the smooth_maximum of the 1000 predictions.
    The idea comes from here: https://arxiv.org/abs/1610.02501, https://arxiv.org/abs/1609.07257
    """

    def __init__(self, mono_model):
        super().__init__(mono_model)
        self.compute_name()

    def forward(self, x):
        mono_model_predictions = self.mono_model(x)
        return mono_model_predictions.mean(axis=-1)
