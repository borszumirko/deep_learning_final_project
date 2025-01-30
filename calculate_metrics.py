import os
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
from models import Generator


print(torch.cuda.device_count())  # Should print 2
print(torch.cuda.get_device_name(0))  # Name of GPU 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def load_images_from_folder(folder_path, max_images=1000, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    images = []
    for filename in os.listdir(folder_path)[:max_images]:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # Convert to uint8 and scale to [0, 255]
                img_tensor = (img_tensor * 255).to(torch.uint8)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return torch.stack(images)

def compute_fid(generator, real_images_path, num_generated_images=1000, latent_dim=100, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()


    # Generate fake images in batches
    fake_images = []
    with torch.no_grad():
        for i in tqdm(range(0, num_generated_images, batch_size), desc="Calculating FID"):
            current_batch_size = min(batch_size, num_generated_images - i)  # Adjust last batch size
            z = torch.randn(current_batch_size, latent_dim, device=device)  # Latent vector batch
            fake_batch = generator(z).to(device)  # Generate images

            # Normalize and convert to uint8
            fake_batch = ((fake_batch + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

            # Load real images
            real_images_batch = load_images_from_folder(real_images_path, max_images=batch_size).to(device)

            # Initialize FID
            fid = FrechetInceptionDistance(feature=2048).to(device)

            # Update FID with real images
            fid.update(real_images_batch, real=True)
            # Update FID with this batch
            fid.update(fake_batch, real=False)

    return fid.compute().item()


num_imgs = 25000

generator1 = Generator(100, 64, 3).to(device)
generator1.load_state_dict(torch.load('trained_models/generator_DCGAN.pth', map_location=device))
fid_score1 = compute_fid(generator1, 'img_align_celeba', num_generated_images=num_imgs)

generator2 = Generator(100, 64, 3).to(device)
generator2.load_state_dict(torch.load('trained_models/generator_WGANGP.pth', map_location=device))
fid_score2 = compute_fid(generator2, 'img_align_celeba', num_generated_images=num_imgs)

generator3 = Generator(100, 64, 3).to(device)
generator3.load_state_dict(torch.load('trained_models/generator_WGANGP_HQ.pth', map_location=device))
fid_score3 = compute_fid(generator3, 'celeba_hq_256', num_generated_images=num_imgs)


print(f"FID Score: {fid_score1}")
print(f"FID Score: {fid_score2}")
print(f"FID Score: {fid_score3}")