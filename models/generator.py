################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""
File to place any kind of generative models
and their respective helper functions.
"""

from abc import abstractmethod
from matplotlib import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from avalanche.models.utils import MLP, Flatten
from avalanche.models.base_model import BaseModel

class AEMLPEncoder(nn.Module):
    """
    Encoder part of the VAE, computer the latent represenations of the input.
    :param shape: Shape of the input to the network: (channels, height, width)
    :param latent_dim: Dimension of last hidden layer
    """

    def __init__(self, shape, latent_dim=128):
        super(AEMLPEncoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.encode = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            MLP([400, latent_dim]),
        )

    def forward(self, x, y=None):
        x = self.encode(x)
        return x


class AEMLPDecoder(nn.Module):
    """
    Decoder part of the VAE. Reverses Encoder.
    :param shape: Shape of output: (channels, height, width).
    :param nhid: Dimension of input.
    """

    def __init__(self, shape, latent_dim=128):
        super(AEMLPDecoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.shape = shape
        self.decode = nn.Sequential(
            MLP([latent_dim, 64, 128, 256, flattened_size], last_activation=False),
            nn.Sigmoid(),
        )


    def forward(self, z, y=None):
        if y is None:
            return self.decode(z).view(-1, *self.shape)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape)




class MlpAE(nn.Module):
    def __init__(self, shape,latent_dim=128, n_classes=10, device="cpu"):
        """Variational Auto-Encoder Class"""
        super(MlpAE, self).__init__()
        # Encoding Layers
        #e_hidden = 128  # Number of hidden units in the encoder. See AEVB paper page 7, section "Marginal Likelihood"
        self.latent_dim = latent_dim
        self.encoder = AEMLPEncoder(shape, latent_dim)
        #self.e_hidden2mean = MLP([e_hidden, latent_dim], last_activation=False)
        #self.e_hidden2logvar = MLP([e_hidden, latent_dim], last_activation=False)

        # Decoding Layers
        self.decoder = AEMLPDecoder(shape, latent_dim)

    def forward(self, x):
        # Shape Flatten image to [batch_size, input_features]
        #x = x.view(-1, 784)

        # Feed x into Encoder to obtain mean and logvar
        z = F.relu(self.encoder(x))
        # mu, logvar = self.e_hidden2mean(x), self.e_hidden2logvar(x)

        # # Sample z from latent space using mu and logvar
        # if self.training:
        #     z = torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add_(mu)
        # else:
        #     z = mu

        # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
        x_recon = self.decoder(z)

        return x_recon

MSE_loss = nn.MSELoss(reduction="sum")
def AE_LOSS(image, reconstruction):
    """Loss for the Variational AutoEncoder."""
    # Binary Cross Entropy for batch
    #BCE = F.binary_cross_entropy(input=reconstruction.view(-1, 28 * 28), target=image.view(-1, 28 * 28),
                                 #reduction='sum')
    BCE = MSE_loss(reconstruction, image)/image.shape[0]
    # Closed-form KL Divergence
    #KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE


__all__ = ["MlpAE", "AE_LOSS"]