import torch
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from torcheval.metrics import FrechetInceptionDistance
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import Dataset
from src.part2.main import make_mnist_data
from src.part2.batched_samples_dataset import BatchSampledImagesDataset, make_mnist_data_sampled
import os
from torchvision import datasets, transforms
from tqdm import tqdm

def preprocess_images_for_inception(images):
    # Reshape images to 2D
    images_2d = images.view(-1, 1, 28, 28)  # From [100, 784] to [100, 1, 28, 28]

    # Convert to 3-channel images by repeating the single channel
    images_3d = images_2d.repeat(1, 3, 1, 1)  # From [100, 1, 28, 28] to [100, 3, 28, 28]

    # Resize to 299x299 as required by Inception v3
    images_resized = TF.resize(images_3d, [299, 299])

    return images_resized

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load real and generated images test data loader and the sampled images data loader
    _, real_images_dataloader = make_mnist_data(batch_size=100, do_transform=False) 
    generated_images_dataloader = make_mnist_data_sampled(batch_size=100, device=device)
    # _, generated_images_dataloader = make_mnist_data(batch_size=100, do_transform=False)

    # Load the pretrained Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception_model.fc = torch.nn.Identity() 
    inception_model.eval()

       # Initialize the FID metric with the Inception V3 model as the feature extractor
    fid_metric = FrechetInceptionDistance( device=device)

    with torch.no_grad():
       
        # Process generated images with tqdm progress bar
        for generated_images in tqdm(generated_images_dataloader, desc="Processing Generated Images"):
            if isinstance(generated_images, (list, tuple)):
                generated_images = generated_images[0]
            generated_images_preprocessed = preprocess_images_for_inception(generated_images)
            # Update FID metric with preprocessed generated images
            fid_metric.update(generated_images_preprocessed, is_real=False)

         # Process real images with tqdm progress bar
        for real_images in tqdm(real_images_dataloader, desc="Processing Real Images"):
            if isinstance(real_images, (list, tuple)):
                real_images = real_images[0]
            real_images_preprocessed = preprocess_images_for_inception(real_images)

            fid_metric.update(real_images_preprocessed, is_real=True)


    # Compute FID score
    fid_score = fid_metric.compute()
    print(f"FID Score: {fid_score}")

