
import torch
import os 
from torch.utils.data import Dataset

from torchvision import transforms

class BatchSampledImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.batch_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]
        self.transform = transform  # Store the transformation
        self.current_batch = None
        self.current_batch_idx = 0
        self.load_next_batch()

    def load_next_batch(self):
        if self.current_batch_idx < len(self.batch_files):
            self.current_batch = torch.load(self.batch_files[self.current_batch_idx])
            self.current_batch_idx += 1

    def __len__(self):
        # Assuming all batches have the same size except possibly the last one
        if self.current_batch is not None:
            return len(self.batch_files) * self.current_batch.size(0)
        return 0


    def __getitem__(self, idx):
        if self.current_batch is None:
            raise IndexError("Index out of range")

        batch_idx = idx % self.current_batch.size(0)
        # Your existing code to load an image...
        image = self.current_batch[batch_idx]

        # If your images are flattened, reshape them here
        if image.dim() == 1:  # Assuming flattened images
            image = image.view(1, 28, 28)  # Reshape from (784,) to (1, 28, 28)

        if self.transform:
            image = self.transform(image)

        return image
    
    # def __getitem__(self, idx):
    #     if self.current_batch is None:
    #         raise IndexError("Index out of range")

    #     batch_idx = idx % self.current_batch.size(0)
    #     image = self.current_batch[batch_idx]

    #     if self.transform:
    #         image = self.transform(image)  # Apply the transformation

    #     if batch_idx == self.current_batch.size(0) - 1 and self.current_batch_idx < len(self.batch_files):
    #         self.load_next_batch()

    #     return image


def make_mnist_data_sampled(data_dir='samples/flow/batch_samples', batch_size=100, device='cpu'):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Lambda(lambda x: x.flatten())
    ])

    # Load the MNIST dataset with transformations
    mnist_train = BatchSampledImagesDataset(data_dir, transform=transform)

    # Use DataLoader to handle sampling
    data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    return data_loader

if __name__ == "__main__":
    data = BatchSampledImagesDataset('samples/flow/batch_samples')

    dataloader = make_mnist_data_sampled(batch_size=32)

    # load into a dataloader
    # dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    # iterate over the dataloader
    for batch in dataloader:
        # plot some images from the batch

        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8,8))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.imshow(np.array(batch[i].cpu().view(28,28)), cmap='gray')
            plt.axis('off')
        plt.show()  

        break

    
