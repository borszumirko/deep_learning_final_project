import torch
from training import train_DCGAN, train_WGAN_GP
from load_dataset import get_DataLoader
from models import Generator, Generator128, Discriminator, Discriminator128, Critic, Critic128
import numpy as np


def run_gan(dataset_path, results_filename, num_epochs, images_save_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # Hyperparameters
    latent_dim = 100  # Dimension of noise vector
    image_size = 64  # CelebA images resized to 64x64
    channels = 3  # RGB images
    batch_size = 256
    lr = 0.0002
    betas = (0.5, 0.999) # for Adam
    epochs = num_epochs

    # Hyperparameters for WGAN-GP
    c_lambda = 10
    crit_repeats = 5
    lr_WGAN_GP = 0.0001
    betas_WGAN_GP = (0, 0.9)



    path = dataset_path

    data_loader = get_DataLoader(path, image_size, batch_size)
    # losses_generator, losses_discriminator = train_DCGAN(latent_dim, image_size, channels, device, epochs, data_loader, lr, betas)
    losses_generator, losses_discriminator = train_WGAN_GP(latent_dim, image_size, channels, device, epochs, data_loader, lr_WGAN_GP, betas_WGAN_GP, c_lambda, crit_repeats, Generator, Critic, images_save_folder)
    losses_discriminator = np.array(losses_discriminator)
    losses_generator = np.array(losses_generator)

    combined = np.column_stack((losses_discriminator, losses_generator))

    np.savetxt(results_filename, combined, fmt='%.2f', delimiter=',', header='dicriminator,generator', comments='')

    print(f"Arrays have been written to {results_filename}")



if __name__=="__main__":
    run_gan('img_align_celeba', 'celeba_losses.txt', 80, 'generated_images_celeba')
    # run_gan('celeba_hq_256', 'celeba_hq_losses.txt', 560, 'generated_images_celeba_hq')