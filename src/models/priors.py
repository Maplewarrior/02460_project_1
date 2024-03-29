import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from src.models.flow import MaskedCouplingLayer, Flow

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

class FlowPrior(nn.Module):
    def __init__(self, masks: torch.tensor, n_transformations: int, latent_dim: int, device: str):
        super(FlowPrior,  self).__init__()
        self.masks = masks.to(device)
        self.M = self.masks.size(1)
        self.latent_dim = latent_dim # latent dimension of scale and translation networks
        self.device = device
        base = GaussianPrior(self.M)
        transformations = self.compose_transformations(n_transformations)
        self.flow_model = Flow(base, transformations)
        # self.__load_state_dict() # load state dict for trained exercise 2.4 model
    
    def compose_transformations(self, n_transformations: int):
        transformations = []
        for i in range(n_transformations):
            scale_net = nn.Sequential(nn.Linear(self.M, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(self.M, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, self.masks[i]))

        return transformations
    
    def sample(self, n_samples):
        # n_samples = torch.Size([num_samples])
        #with torch.no_grad():
        samples = self.flow_model.sample(n_samples)
        return samples
    
    def log_prob(self, ipt):
        # with torch.no_grad():
        log_prob = self.flow_model.log_prob(ipt)
        return log_prob
    
    def forward(self):
        return self.flow_model
    
# Variational mixture of posteriors prior using pseudo-inputs
class VampPrior(nn.Module):
    def __init__(self, num_components, D, encoder):
        """
        Define a VampPrior distribution.

        Parameters:
        num_components: [int]
            The number of Gaussian components in the mixture.
        latent_dim: [int]
            Dimension of the latent space.
        num_pseudo_inputs: [int]
            Number of pseudo inputs to use in the VampPrior.
        """
        super(VampPrior, self).__init__()
        self.num_components = num_components
        self.D = D
        self.encoder = encoder

        # Parameters for mixture weights (pi) - learnable
        self.mixture_weights = nn.Parameter(torch.randn(self.num_components), requires_grad=True)
        # Parameters for pseudo inputs
        self.pseudo_inputs = nn.Parameter(torch.rand(self.num_components, self.D), requires_grad=True)

    def forward(self):
        """
        Return the VampPrior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # Use softmax to ensure the mixture weights sum to 1 and are non-negative
        mixture_weights = torch.nn.functional.softmax(self.mixture_weights, dim=0)
        # Create a categorical distribution for selecting the components
        categorical = td.Categorical(probs=mixture_weights)
        component_distribution = self.encoder(self.pseudo_inputs)
        # Combine the above into a MixtureSameFamily distribution
        prior = td.MixtureSameFamily(categorical, component_distribution)
        return prior
    