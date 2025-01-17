import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('gan_loss.csv')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['D_loss'], label='Discrimintor loss', color='blue', marker='o')
plt.plot(data['G_loss'], label='Generator loss', color='orange', marker='x')

# Customize the plot
plt.title('Discriminator and Generator Losses Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('BCE Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
