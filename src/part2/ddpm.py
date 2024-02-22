# Credits for Code temmplate to Jes Frellsen, 2024

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        # ? x0~q(x0) 
        # ! received as input x
        
        # ? t~Unif({1,...,T})
        t = torch.randint(0, self.T, size=(x.shape[0], 1), device=x.device)
        alpha_bar = self.alpha_cumprod[t]

        # ? eps~N(0,I)
        eps = torch.randn_like(x)
         
        # ? E_{t,x0,eps}[|| eps - eps_theta(sqrt(alpha_bar)·x0 + sqrt(1-alpha_bar)·eps, t) ||^2]
        eps_theta = self.network(torch.sqrt(alpha_bar)*x + torch.sqrt(1-alpha_bar)*eps, t) # note t should be normalized!
        # neg_elbo = torch.norm(eps - eps_theta, p=2, dim=1)**2 # ? works for tg and cb
        neg_elbo = F.mse_loss(eps_theta, eps, reduction='none') # F.mse_loss more numerically stable?

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            z = (torch.randn(shape) if t > 0 \
                    else torch.zeros(shape)).to(self.alpha.device)
            alpha = self.alpha[t]
            alpha_bar = self.alpha_cumprod[t]
            _t = torch.tensor(t).expand((x_t.shape[0], 1)).to(x_t.device) / self.T # note t is normalized!
            eps_theta = self.network(x_t, _t) 
            sigma_t = torch.sqrt(self.beta[t])
            mu = (x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar)*eps_theta) / torch.sqrt(alpha)
            
            x_t = mu + sigma_t * z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()
