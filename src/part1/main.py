import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

from src.models.vae_bernoulli import VAE, BernoulliDecoder, GaussianEncoder, make_enc_dec_networks
from src.models.priors import FlowPrior, MixtureOfGaussiansPrior, GaussianPrior, VampPrior
from src.models.flow import create_mask


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
        losses.append(run_losses)
    
    # compute mean and std
    mean_loss = torch.mean(torch.tensor(losses))
    std_loss = torch.std(torch.tensor(losses))
    return mean_loss.item(), std_loss.item(), losses

def make_evaluation_results(config, model, data_loader, n_runs):
    _, _, losses = evaluate_runs(model, data_loader, config['device'], n_runs)
    # log metadata
    results = {k: v for k, v in config.items() if k != 'mode'}
    # log losses for each run
    for i, run in enumerate(losses):
        results[f'run_{i+1}'] = run
    
    # save results
        ### TODO: Update filename with loss type (ELBO, IWAE) used for evaluation
    modelname = config['model'].strip('.pt')
    lossname = "ELBO" if config['k'] == 1 else "IWAE"
    with open(f'results/{modelname}_{lossname}_eval.json', 'w') as f:
        json.dump(results, f)


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

def sample_posterior(model, data_loader, device, num_samples=10000):
    #model.eval()
    posterior_samples = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            # posterior_samples.append(z.cpu().numpy())
            # labels.append(y.cpu().numpy())
            batch_size = x.size(0)

            # Randomly select num_samples from the current batch
            indices = random.sample(range(batch_size), min(num_samples, batch_size))
            posterior_samples.append(z[indices].cpu().numpy())
            labels.append(y[indices].cpu().numpy())
            
            # Check if we have collected enough samples
            if len(posterior_samples) * batch_size >= num_samples:
                break
            
    posterior_samples = np.concatenate(posterior_samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    return posterior_samples, labels

def prior_posterior_plot(model, data_loader, device, args):
    """
    Creates a contour plot of the prior and superimposes posterior samples.
    """
    # Sample from the posterior
    posterior_samples, labels = sample_posterior(model, data_loader, device)

    # If dimensions are greater than 2, apply PCA
    if posterior_samples.shape[1] > 2:
        pca = PCA(n_components=2)
        posterior_samples = pca.fit_transform(posterior_samples)
        
    # Create a meshgrid with latent dim = 2
    nx, ny = (600, 600)
    # set coords to min and max of the samples
    coords = np.max(np.abs(posterior_samples))
    #coords = 9

    x = np.linspace(-coords, coords, nx)
    y = np.linspace(-coords, coords, ny)
    xv, yv = np.meshgrid(x, y)
    meshgrid = np.stack((xv.flatten(), yv.flatten()), axis=1)
    mesh_tensor = torch.FloatTensor(meshgrid).to(device) #shape is (nx*ny, 2) = torch.size([10000, 2])
    log_prior_density = model.prior().log_prob(mesh_tensor).detach().cpu().numpy()

     
    # Create a contour plot
    plt.figure(figsize=(14, 14))
    contour = plt.contourf(x, y, log_prior_density.reshape(nx, ny), cmap='viridis', levels=60, alpha=0.7)
    scatter = plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], c=labels, cmap='tab10', s=1, alpha=0.8)

    handles, labels = scatter.legend_elements(num=10)
    legend = plt.legend(handles, labels, title="Class Labels", loc='upper right')
    colorbar = plt.colorbar(contour, label="Log Prior Density", fraction=0.03, pad=0.01)

    # Adjust figure size to accommodate colorbar on the right
    plt.subplots_adjust(right=0.85)

    priorname = args.prior.replace('_', ' ')
    plt.title(f'Contour plot of Log {priorname} Prior Density with Posterior samples', fontsize=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust rect for title and ylabel
    plt.savefig(f'results/prior_posterior_{priorname}.png')
    plt.show()

    # plt.figure(figsize=(14, 14))
    # contour = plt.contourf(x, y, log_prior_density.reshape(nx, ny), cmap='viridis', levels=60, alpha=0.7)
    # scatter = plt.scatter(posterior_samples[:, 0], posterior_samples[:, 1], c=labels, cmap='tab10', s=1, alpha=0.8)

    # handles, labels = scatter.legend_elements(num=10)
    # legend = plt.legend(handles, labels, title="Class Labels", loc='upper right', bbox_to_anchor=(1.2, 1))
    # colorbar = plt.colorbar(contour, label="Log Prior Density", fraction=0.1, pad=0.01)
    # priorname = args.prior.replace('_', ' ')
    # plt.title(f'Contour plot of Log {priorname} Prior Density with Posterior samples')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.tight_layout(rect=[0, 0, 0.98, 1]) # Rect(left, bottom, right, top)
    # plt.show()

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
    parser.add_argument('--prior', type=str, default='standard_normal', choices=['Standard_Normal', 'MoG', 'Flow', 'Vamp'], help='Type of prior distribution over latents e.g. p(z)')
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'chequerboard'], help='Type of mask to use with flow prior (default: %(default)s)')
    parser.add_argument('--k', type=int, default=1, help='The sample size when using IWAE loss (default: %(default)s)')
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
    if args.prior == 'Standard_Normal':
        prior = GaussianPrior(M)
    elif args.prior == 'MoG':
        prior = MixtureOfGaussiansPrior(latent_dim=M, num_components=10)
    elif args.prior == 'Flow':
        mask = create_mask(M=M, mask_type='random')
        pdb.set_trace()
        prior = FlowPrior(mask=mask, n_transformations=20, latent_dim=256, device=args.device)
    elif args.prior == 'Vamp':
        prior = VampPrior(num_components=50, latent_dim=M, num_pseudo_inputs=500)


    # Define VAE model
    encoder_net, decoder_net = make_enc_dec_networks(M)
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder, args.k).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
        make_evaluation_results(vars(args), model, mnist_test_loader, n_runs=10)
        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        pdb.set_trace()
        evaluate_runs(model, mnist_test_loader, args.device)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            #samples = model.sample(1)
            #samples = (model.sample(64)).cpu() 
            ##save_image(samples.view(64, 1, 28, 28), args.samples)

            # Evaluate the ELBO on the test set
            #elbos = model.get_elbos(mnist_test_loader, device)
            #print(f"ELBO on test set: {torch.stack(elbos).mean():.1f}")
            
            #Plot posterior samples
            prior_posterior_plot(model, mnist_test_loader, device, args)


