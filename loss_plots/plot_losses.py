import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv('celeba_DCGAN copy.csv')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['discriminator'], label='Discriminator loss', color='blue')
plt.plot(data['generator'], label='Generator loss', color='orange')

# Customize the plot
plt.title('Discriminator and Generator Losses Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Wasserstein loss with gradient penalty', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('DCGAN_losses_2.pdf', format='pdf')

# Show the plot
plt.show()
