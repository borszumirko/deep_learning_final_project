import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms



class CelebAFlatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # List all image files in the directory
        self.image_files = [f for f in os.listdir(root) if f.lower().endswith(("jpg", "png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            img = self.transform(img)
        return img, 0  # Return a dummy label (0) to match DataLoader expectations

def get_DataLoader(path, image_size, batch_size):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])


    local_celeba_path = path

    celeba_dataset = CelebAFlatDataset(root=local_celeba_path, transform=transform)

    # celeba_dataset = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    print('dataset loaded...')
    return data_loader