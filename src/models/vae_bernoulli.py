import torch
import torch.nn as nn
import torch.distributions as td
import pdb

def make_enc_dec_networks(M: int):
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )
    return encoder_net, decoder_net

def make_enc_dec_networks_cnn(M: int):
    from src.models.utils import Unsqueeze, Reshape

    # Define encoder and decoder networks, CNN type
    encoder_net = nn.Sequential(
        Unsqueeze(dim=1),
        nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(output_size=1),
        Reshape((64,)),
        nn.Linear(64, 128),
        nn.Linear(128, M*2)
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 64),
        Reshape((64, 1, 1)), 
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4,stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4,stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4,stride=2, padding=3),
        nn.Sigmoid()
    )

    return encoder_net, decoder_net

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)
        
class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class MultivariateGaussianDecoder(nn.Module):
    def __init__(self, decoder_net, learn_variance=False):
        """
        Define a multivariate gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(MultivariateGaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.learn_variance = learn_variance
        self.log_var = nn.Parameter(torch.log(torch.ones(28, 28) * .5), requires_grad=learn_variance)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where `M` is the dimension of the latent space.
        """
        # pdb.set_trace()
        dec_out = self.decoder_net(z)
        std = torch.exp(.5 * self.log_var)
        return td.Independent(td.Normal(loc=dec_out, scale=std), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, k=1):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space. p(x|z)
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space. --> p(z|x)
        k: [int]
            The number of samples to use for IWAE. If set to 1, this is equivalent to using the ELBO loss.
            Note: Reparamterization trick is used and therefore estimates will vary for the same x.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.k = k
        

    def iwae(self, x):
        """
        Compute the IWAE for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        x = x.repeat(self.k, 1, 1)
        q = self.encoder(x)
        h = q.rsample()

        log_p_x_h = self.decoder(h).log_prob(x)
        log_p_h = self.prior().log_prob(h) 
        log_q_h_x = q.log_prob(h)
        marginal_likelihood = (log_p_x_h + log_p_h - log_q_h_x).view(self.k, x.size(0)//self.k)

        iwae = (torch.logsumexp(marginal_likelihood, dim=0) - torch.log(torch.tensor(self.k))).mean()
        
        return iwae


    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample() # reparameterization
        RE = self.decoder(z).log_prob(x)
        KL = q.log_prob(z) - self.prior().log_prob(z)
        elbo = (RE - KL).mean()
        """
        Original code: 
        elbo_old = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        """
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def loss(self, x):
        """
        Lazy way to change interface to match part2/main.py
        """
        return self(x)

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x) if self.k == 1 else -self.iwae(x)