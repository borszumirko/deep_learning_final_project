import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import Generator, Discriminator

def train(latent_dim, image_size, channels, device, epochs, data_loader, lr, betas):
    # Initialize models
    generator = Generator(latent_dim, image_size, channels).to(device)
    discriminator = Discriminator(image_size, channels).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # Save generated images
    constant_noise = torch.randn(25, latent_dim).to(device)
    def save_generated_images(epoch, generator):
        fake_images = generator(constant_noise).detach().cpu()
        fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]
        
        fake_images = fake_images[:25]
        grid = torch.cat([torch.cat([fake_images[i * 5 + j] for j in range(5)], dim=2) for i in range(5)], dim=1)

        # Display the grid
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f"Epoch {epoch}")
        os.makedirs("generated_images", exist_ok=True)
        plt.savefig(f"generated_images/celebA_epoch_{epoch}.png")
        # plt.show()

    losses_generator = []
    losses_discriminator = []

    # Training loop
    for epoch in range(epochs):
        if (epoch) % 5 == 0:
            save_generated_images(epoch, generator)
        
        l_generator = 0.0
        l_discriminator = 0.0
        num_batches = 0

        for real_images, _ in tqdm(data_loader, desc="Training"):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)


            real_loss = criterion(discriminator(real_images), real_labels)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_loss = criterion(discriminator(fake_images), real_labels)  # Fool the discriminator

            optimizer_g.zero_grad()
            fake_loss.backward()
            optimizer_g.step()

            l_generator += fake_loss.item()
            l_discriminator += d_loss.item()
            num_batches += 1

        l_generator /= num_batches
        l_discriminator /= num_batches

        # Print losses and save images
        print(f"Epoch [{epoch + 1}/{epochs}] D Loss: {l_discriminator:.4f}, G Loss: {l_generator:.4f}")
        if epoch == (epochs - 1):
            save_generated_images(epoch + 1, generator)

        losses_discriminator.append(l_discriminator)
        losses_generator.append(l_generator)

    return losses_generator, losses_discriminator