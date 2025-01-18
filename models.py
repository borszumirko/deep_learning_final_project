import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size, image_size, channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # Out: 1024 x 4 x 4

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Out: 512 x 8 x 8

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Out: 256 x 16 x 16

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Out: 128 x 32 x 32

            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output is between -1 to 1
            # Out: 3 x 64 x 64
        )

    def forward(self, z):
        # Reshape z to have shape (batch_size, latent_size, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: 3 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128 x 32 x 32 -> 256 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 256 x 16 x 16 -> 512 x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 512 x 8 x 8 -> 1024 x 4 x 4
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid()
        )


    def forward(self, img):
        out = self.model(img)
        return out


class Critic(nn.Module):
    def __init__(self, image_size, channels):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: 3 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128 x 32 x 32 -> 256 x 16 x 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 256 x 16 x 16 -> 512 x 8 x 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 512 x 8 x 8 -> 1024 x 4 x 4
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
        )


    def forward(self, img):
        out = self.model(img)
        return out