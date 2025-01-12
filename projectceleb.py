import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100  # Dimension of noise vector
image_size = 64  # CelebA images resized to 64x64
channels = 3  # RGB images
hidden_dim = 256
batch_size = 128
epochs = 50

# CelebA Dataset (download and preprocess)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

celeba_dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)


# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Generator, self).__init__()
        self.init_size = image_size // 4  # Initial size after upscaling
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),  # Output scaled to [-1, 1]
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.model(out)
        return img


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 4) ** 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Initialize models
generator = Generator(latent_dim, image_size, channels).to(device)
discriminator = Discriminator(image_size, channels).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Save generated images
def save_generated_images(epoch, generator):
    z = torch.randn(16, latent_dim).to(device)
    fake_images = generator(z).detach().cpu()
    fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]
    grid = torch.cat([fake_images[i] for i in range(16)], dim=2)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Epoch {epoch}")
    os.makedirs("generated_images", exist_ok=True)
    plt.savefig(f"generated_images/celebA_epoch_{epoch}.png")
    plt.show()


# Training loop
for epoch in range(epochs):
    for real_images, _ in data_loader:
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

    # Print losses and save images
    print(f"Epoch [{epoch + 1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {fake_loss.item():.4f}")
    if (epoch + 1) % 5 == 0:
        save_generated_images(epoch + 1, generator)
