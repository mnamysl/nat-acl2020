import warnings
from pathlib import Path

import torch.nn

from abc import abstractmethod

from typing import Union, List

import flair
from flair.data import Sentence
from flair.training_utils import Result
from flair.nn import Model


class ParameterizedModel(Model):
    """Abstract base class for all downstream task models in Flair, such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    @abstractmethod
    def forward_loss(self, sentences: Union[List[Sentence], Sentence], params: dict = {}) -> (torch.tensor, dict):
        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""
        pass
