import torch
import numpy as np
from training import train
from load_dataset import get_DataLoader

def run_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # Hyperparameters
    latent_dim = 100  # Dimension of noise vector
    image_size = 64  # CelebA images resized to 64x64
    channels = 3  # RGB images
    hidden_dim = 256
    batch_size = 256
    lr = 0.0002
    betas = (0.5, 0.999) # for Adam
    epochs = 60

    path = 'img_align_celeba'

    data_loader = get_DataLoader(path, image_size, batch_size)
    losses_generator, losses_discriminator = train(latent_dim, image_size, channels, device, epochs, data_loader, lr, betas)

    losses_discriminator = np.array(losses_discriminator)
    losses_generator = np.array(losses_generator)

    combined = np.column_stack((losses_discriminator, losses_generator))

    np.savetxt('losses.txt', combined, fmt='%.2f', delimiter=',', header='discriminator,generator', comments='')

    print("Arrays have been written to 'losses.txt'")




if __name__=="__main__":
    run_gan()