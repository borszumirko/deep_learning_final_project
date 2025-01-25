import torch
import matplotlib.pyplot as plt
import numpy as np
from models import Generator

generator = Generator(100, 3, 64)
generator2 = Generator(100, 3, 256)

generator.load_state_dict(torch.load('generator_celeba_80.pth', map_location=torch.device('cpu')))
generator2.load_state_dict(torch.load('generator_celeba_hq_560.pth', map_location=torch.device('cpu')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator2.to(device)

batch_size = 16
z = torch.randn(batch_size, 100, 1, 1, device=device)  # Noise vector

with torch.no_grad():
    generator.eval()
    fake_images = generator(z)
    fake_images2 = generator2(z)

fake_images = (fake_images * 0.5 + 0.5).cpu()  # Rescale from [-1, 1] to [0, 1]
fake_images2 = (fake_images2 * 0.5 + 0.5).cpu()


grid_size = int(np.sqrt(batch_size))
fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(10, 10),
                         gridspec_kw={'wspace': 0, 'hspace': 0})

for i, ax in enumerate(axes.flatten()):
    if i < batch_size:
        ax.axis("off")
        ax.imshow(np.transpose(fake_images[i].numpy(), (1, 2, 0)))
    else:
        ax.axis("off")
        ax.imshow(np.transpose(fake_images2[i - batch_size].numpy(), (1, 2, 0)))
    
plt.subplots_adjust(wspace=0, hspace=0)  # Remove spacing between subplots
plt.show()
