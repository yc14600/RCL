################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import torch.nn as nn
import torch
from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)
from avalanche.models.base_model import BaseModel
from avalanche.models.generator import MLP, Flatten

class encoder_model(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    **Example**::
        >>> from avalanche.models import SimpleMLP
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleMLP(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        shape = (1, 28, 28),
        latent_dim = 128
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        flattened_size = torch.Size(shape).numel()
        #latent_dim = 128
        layers = nn.Sequential(
            *(
                Flatten(),
                nn.Linear(in_features=flattened_size, out_features=400),
                nn.BatchNorm1d(400),
                nn.LeakyReLU(),
                MLP([400, latent_dim]),
            )
        )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x


__all__ = ["encoder_model"]