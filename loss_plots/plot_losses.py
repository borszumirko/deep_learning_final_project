import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('celeba_DCGAN copy.csv')

plt.figure(figsize=(10, 6))
plt.plot(data['discriminator'], label='Discriminator loss', color='blue')
plt.plot(data['generator'], label='Generator loss', color='orange')

plt.title('Discriminator and Generator Losses Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Wasserstein loss with gradient penalty', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig('DCGAN_losses_2.pdf', format='pdf')

plt.show()
