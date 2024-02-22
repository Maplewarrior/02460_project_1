import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from part2.flow import MaskedCouplingLayer, Flow

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
    def __init__(self, M, n_transformations: int, latent_dim: int, device: str):
        super(FlowPrior,  self).__init__()
        self.M = M
        self.latent_dim = latent_dim
        self.mask = self.build_mask()
        self.device = device
        base = GaussianPrior(self.M)
        transformations = self.compose_transformations(n_transformations)
        self.flow_model = Flow(base, transformations)
        # self.__load_state_dict() # load state dict for trained exercise 2.4 model
    
    def compose_transformations(self, n_transformations: int):
        """
        Tanh is not added because the model was not trained with MNIST (see exercises for week 2)
        """
        transformations = []
        for i in range(n_transformations):
            mask_inv = (1-self.mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(self.M, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.M))
            translation_net = nn.Sequential(nn.Linear(self.M, self.latent_dim), nn.ReLU(), nn.Linear(self.latent_dim, self.M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask_inv))

        return transformations

    def build_mask(self):
        """
        Default mask from week 2
        """
        mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
        mask = torch.zeros((self.M,))
        mask[self.M//2:] = 1
        return mask
    
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