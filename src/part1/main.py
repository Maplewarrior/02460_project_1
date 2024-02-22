import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from vae_bernoulli import VAE, BernoulliDecoder, GaussianEncoder
from priors.priors import FlowPrior, MixtureOfGaussiansPrior, GaussianPrior

def create_mask(D: int = 784, mask_type: str = 'random'):
    """
    Function for creating a mask for a flow based prior
        @param D: The dimensionality of the mask --> needs to match latent dim of VAE.
        @param mask_type: Options are [chequerboard, random]
    """
    if mask_type == 'random': # create a random mask with 50% ones and rest 0
        mask = torch.zeros((D))
        mask[D//2:] = 1
        mask = mask[torch.randperm(len(mask))]
    
    elif mask_type == 'chequerboard':
        mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(D//2) for j in range(D//2)])

    return mask


def evaluate(model, data_loader, device):
    """
    Evaluates a VAE model.

    Parameters:
    model: [VAE]
        The VAE model to evaluate
    data_loader: [DataLoader]
        Iterable, the dataloader on which the model should be evaluated.
    device: [torch.device]
        The device on which evaluation should be run.
    
    Returns:
    losses: [List]
        A list containing losses for each batch.
    """
    model.eval()
    num_steps = len(data_loader)
    losses = []
    with torch.no_grad():
        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                x = next(iter(data_loader))[0]
                x = x.to(device)
                
                loss = model(x)
                losses.append(loss.item())
                # Report
                if step % 5 ==0 :
                    loss = loss.detach().cpu()
                    pbar.set_description(f"step={step}, loss={loss:.1f}")
     
    print(f'Mean loss of model on MNIST test: {torch.mean(torch.tensor(losses)):.4f}')
    return losses

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader)*epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1

def prior_posterior_plot():
    """
    Creates a contour plot of the prior and superimposes poster samples.
    """
    pass


if __name__ == '__main__':
    import pdb

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'sample_posterior'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='standard_normal', choices=['standard_normal', 'MoG', 'flow'], help='Type of prior distribution over latents e.g. p(z)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'standard_normal':
        prior = GaussianPrior(M)
    elif args.prior == 'MoG':
        prior = MixtureOfGaussiansPrior(M, K=10)
    elif args.prior == 'flow':
        prior = FlowPrior(M=M, n_transformations=5, latent_dim=args.latent_dim, device=args.device)

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

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)
