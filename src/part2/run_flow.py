# Credits for Code temmplate to Jes Frellsen, 2024

import torch
import torch.nn as nn
from tqdm import tqdm
from src.part2.flow import MaskedCouplingLayer, Flow, GaussianBase


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
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

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def make_flow_model(D, mask_type = "default", device = "cpu", num_transformations = 8, num_hidden = 5):

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

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--mask', type=str, default='default', choices=['random', 'chequerboard', 'default'], help='type of mask to use for the coupling layers (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Generate the data

    # MNIST Dataset loading
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
        transforms.Lambda(lambda x: x.flatten())
    ])

    train_dataset = datasets.MNIST('data/', train=True, download=True, transform=mnist_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = datasets.MNIST('data/', train=False, download=True, transform=mnist_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    first_batch = next(iter(train_loader))
    D = first_batch[0].shape[1]

    model = make_flow_model(D, mask_type=args.mask, device=device, num_transformations=8, num_hidden=5)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train mode
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        from torchvision.utils import save_image
        # import matplotlib.pyplot as plt
        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample((64,))).cpu() 

        save_image(samples.view(64, 1, 28, 28), args.samples)