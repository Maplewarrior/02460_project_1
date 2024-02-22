import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, num_components, latent_dim):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        num_components: [int]
            The number of Gaussian components in the mixture.
        latent_dim: [int]
            Dimension of the latent space.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        # Parameters for mixture weights (pi)
        self.mixture_weights = nn.Parameter(torch.randn(self.num_components), requires_grad=True)

        # Parameters for component means and standard deviations
        self.means = nn.Parameter(torch.randn(self.num_components, self.latent_dim), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.num_components, self.latent_dim), requires_grad=True)

    def forward(self):
        """
        Return the MoG prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # Use softmax to ensure the mixture weights sum to 1 and are non-negative
        mixture_weights = torch.nn.functional.softmax(self.mixture_weights, dim=0)

        # Create a categorical distribution for selecting the components
        categorical = td.Categorical(probs=mixture_weights)

        # Create a distribution for the components using Independent and Normal
        component_distribution = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.stds)), 1)

        # Combine the above into a MixtureSameFamily distribution
        prior = td.MixtureSameFamily(categorical, component_distribution)

        return prior