import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from src.models.vae_bernoulli import VAE, BernoulliDecoder, GaussianEncoder, make_enc_dec_networks
from src.models.priors.priors import FlowPrior, MixtureOfGaussiansPrior, GaussianPrior

def create_mask(M: int = 784, mask_type: str = 'random'):
    """
    Function for creating a mask for a flow based prior
        @param D: The dimensionality of the mask --> needs to match latent dim of VAE.
        @param mask_type: Options are [chequerboard, random]
    """
    if mask_type == 'random': # create a random mask with 50% ones and rest 0
        mask = torch.zeros((M))
        mask[M//2:] = 1
        mask = mask[torch.randperm(len(mask))]
    
    elif mask_type == 'chequerboard':
        mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(M//2) for j in range(M//2)])

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

def evaluate_runs(model, data_loader, device, n_runs):
    losses = []
    # make n_runs evaluations
    for i in range(n_runs):
        run_losses = evaluate(model, data_loader, device)
        losses.extend(run_losses)
    
    # compute mean and std
    mean_loss = torch.mean(torch.tensor(losses))
    std_loss = torch.std(torch.tensor(losses))

    return mean_loss.item(), std_loss.item(), losses


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
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'chequerboard'], help='Type of mask to use with flow prior (default: %(default)s)')
    
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("")

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
        prior = MixtureOfGaussiansPrior(latent_dim=M, num_components=10)
    elif args.prior == 'flow':
        mask = create_mask(M=M, mask_type='random')
        pdb.set_trace()
        prior = FlowPrior(mask=mask, n_transformations=5, latent_dim=256, device=args.device)

    # Define VAE model
    encoder_net, decoder_net = make_enc_dec_networks(M)
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
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        # losses = evaluate(model, mnist_test_loader, args.device)
        mu, std, losses = evaluate_runs(model, mnist_test_loader, args.device, 3)
        pdb.set_trace()
    
