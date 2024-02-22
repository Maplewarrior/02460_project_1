
import torch
import os 
from torch.utils.data import Dataset

class BatchSampledImagesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.batch_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')]
        self.current_batch = None
        self.current_batch_idx = 0
        self.load_next_batch()

    def load_next_batch(self):
        if self.current_batch_idx < len(self.batch_files):
            self.current_batch = torch.load(self.batch_files[self.current_batch_idx])
            self.current_batch_idx += 1

    def __len__(self):
        if self.current_batch is not None:
            return len(self.batch_files) * self.current_batch.size(0)
        return 0

    def __getitem__(self, idx):
        if self.current_batch is None:
            # Handle the case when there are no more batches to load
            raise IndexError("Index out of range")

        batch_idx = idx % self.current_batch.size(0)
        image = self.current_batch[batch_idx]

        # Load the next batch if we've reached the end of the current one and there are more batches to load
        if batch_idx == self.current_batch.size(0) - 1 and self.current_batch_idx < len(self.batch_files):
            self.load_next_batch()

        return image

    

if __name__ == "__main__":
    data = BatchSampledImagesDataset('samples/flow/batch_samples')

    # load into a dataloader
    dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

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

    
