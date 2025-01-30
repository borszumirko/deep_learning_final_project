import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import Generator, Discriminator

# source: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
def get_gradient(crit, real, fake, epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
        
    )[0]
    return gradient

# source: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan
def get_gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)
    
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty



def train_DCGAN(latent_dim, image_size, channels, device, epochs, data_loader, lr, betas):
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
        plt.show()

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


def train_WGAN_GP(latent_dim, image_size, channels, device, epochs, data_loader, lr, betas, c_lambda, critic_repeats, G, D, images_save_folder):

    # Initialize models
    generator = G(latent_dim, image_size, channels).to(device)
    critic = D(image_size, channels).to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_c = optim.Adam(critic.parameters(), lr=lr, betas=betas)

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
        os.makedirs(images_save_folder, exist_ok=True)
        plt.savefig(f"{images_save_folder}/celebA_epoch_{epoch}.png")
        plt.show()

    losses_generator = []
    losses_critic = []

    for epoch in range(epochs):
        if (epoch) % 5 == 0:
            save_generated_images(epoch, generator)
        
        l_generator = 0.0
        l_critic = 0.0
        num_batches = 0

        for real_images, _ in tqdm(data_loader, desc="Training"):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Critic
            mean_crit_loss = 0
            for _ in range(critic_repeats):
                optimizer_c.zero_grad()
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_images = generator(noise)
                # fake_images.detach() is needed to the generator's weights are not updated during the critic's training step
                crit_fake_scores = critic(fake_images.detach())
                crit_real_scores = critic(real_images)

                epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(critic, real_images, fake_images, epsilon)
                gradient_penalty = get_gradient_penalty(gradient)
                # Calculate critic loss
                critic_loss = torch.mean(crit_fake_scores) - torch.mean(crit_real_scores) + c_lambda * gradient_penalty
                mean_crit_loss += critic_loss
                
                # Multiple backward passes on the same graph ==> retain_graph=True
                critic_loss.backward(retain_graph=True)
                
                optimizer_c.step()

            mean_crit_loss /= critic_repeats

            # Train Generator
            optimizer_g.zero_grad()
            noise_train_g = torch.randn(batch_size, latent_dim).to(device)
            fake_images_train_g = generator(noise_train_g)
            crit_fake_scores_train_g = critic(fake_images_train_g)

            # To train the generator we want crit_fake_scores_train_g to be high ('1.' to cast to float)
            generator_loss = -1. * torch.mean(crit_fake_scores_train_g)
            generator_loss.backward()
            optimizer_g.step()


            l_generator += generator_loss.item()
            l_critic += mean_crit_loss.item()
            num_batches += 1

        l_generator /= num_batches
        l_critic /= num_batches

        # Print losses and save images
        print(f"Epoch [{epoch + 1}/{epochs}] C Loss: {l_critic:.4f}, G Loss: {l_generator:.4f}")
        if epoch == (epochs - 1):
            save_generated_images(epoch + 1, generator)

        losses_critic.append(l_critic)
        losses_generator.append(l_generator)

    return losses_generator, losses_critic