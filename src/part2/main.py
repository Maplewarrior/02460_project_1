from tqdm import tqdm
import torch
import torch.nn as nn
from src.part2.flow import MaskedCouplingLayer, Flow, GaussianBase


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    mean_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.detach().item() * x.shape[0]
            # Update progress bar
            progress_bar.set_postfix(loss=f" {loss.item():12.4f}", mean_loss=f" {mean_loss:12.2f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        mean_loss = total_loss / len(data_loader)

def make_mnist_data():
    transform = transforms.Compose([transforms.ToTensor() ,
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x -0.5) * 2.0),
        transforms.Lambda(lambda x: x.flatten())]
        )
    train_data = datasets.MNIST('data/',
        train = True,
        download = True,
        transform = transform
        )
    test_data = datasets.MNIST('data/',
        train = False,
        download = True,
        transform = transform
        )
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=args.batch_size, 
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size=args.batch_size, 
                                                shuffle=False)
    
    return train_loader, test_loader

def make_flow_model(D, mask_type = "default", device = "cpu", num_transformations = 8, num_hidden = 5):
    """ Make a flow model. 
    
    Args:
        D: [int]
            Dimension of the data.
        mask_type: [str]
            Specification of a specific mask. Options are 'random' or 'chequerboard'. 
            The former will randomly permute the mask and the latter will give a chequerboard pattern.
        device: [str]
            The device to use for training.
        num_transformations: [int]
            Number of transformations to use in the flow model.
        num_hidden: [int]
            Number of hidden units in the scaling and translation networks.

    Returns:
        model: [Flow]
            The flow model.
    
    """

    # Define the prior distribution
    base = GaussianBase(D)

    # Define transformations
    transformations =[]
    mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])

    # Make a mask that is 1 for the first half of the features and 0 for the second half
    if mask_type == 'default' or mask_type == 'random':
        mask = torch.zeros((D,))
        mask[D//2:] = 1
    elif mask_type == 'chequerboard':
        mask = torch.zeros((28, 28))
        cheq_size = 2
        # Set 1s in a chequerboard pattern
        mask[0::cheq_size, 0::cheq_size] = 1
        mask[1::cheq_size, 1::cheq_size] = 1 
        mask = mask.flatten()

    mask = mask.to(device)
    
    for i in range(num_transformations):
        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D),nn.Tanh())

        if mask_type == 'random':
            permuted_indices = torch.randperm(D)
            permuted_mask = mask[permuted_indices]
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask=permuted_mask))
        else:
            mask = (1-mask) # Flip the mask
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    # Define flow model
    model = Flow(base, transformations).to(device)
    return model

"""
params: 
@T: number of steps in the diffusion process, default=1_000
"""
def make_ddpm(T = 1_000):
    from src.models.unet import Unet
    from src.models.ddpm import DDPM

    # Define the (mu,sigma2)-estimator network
    network = Unet()

    # Define model
    model = DDPM(network, T=T).to(args.device)
    if args.continue_train == True:
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

    return model

def make_vae():
    # from src.models
    raise Exception("flow model initialization not implemented yet!")

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    
    parser.add_argument('--model-type', type=str, choices=['flow', 'ddpm', 'vae'], help='torch device (default: %(default)s)')
    
    parser.add_argument('--continue-train', type=bool, default=False, help='whether to continue training from ckpt (same path as "model") (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    train_loader, test_loader = make_mnist_data()
    
    # Get the dimension of the dataset
    D = next(iter(train_loader))[0].shape[1]

    if args.model_type == 'flow':
        raise Exception("flow model initialization not implemented yet!")
    elif args.model_type == 'ddpm':
        model = make_ddpm()
    elif args.model_type == 'vae':
        model = make_vae()

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device))) # ret vigtig :)
        model.eval()
        with torch.no_grad():
            n_samples = 4
            samples = (model.sample((n_samples,D))).cpu()
            # Transform the samples back to the original space
            samples = samples / 2 + 0.5
            for i in range(n_samples):
                save_path =  f"{(args.samples).split('.pdf')[0]}_{i}.pdf" # make save_path in format {save_loc}/{filename}_{1,2,...,n_samples}.{ext}
                save_image(samples[i].view(1, 1, 28, 28), save_path, format='pdf')
    