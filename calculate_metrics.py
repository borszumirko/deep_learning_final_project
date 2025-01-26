import os
import torch
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
from tqdm import tqdm
from models import Generator

def load_images_from_folder(folder_path, max_images=1000, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    images = []
    for filename in tqdm(os.listdir(folder_path)[:max_images], desc="Loading real images"):
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

def compute_fid(generator, real_images_path, num_generated_images=1000, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()
    
    # Load real images
    real_images = load_images_from_folder(real_images_path, max_images=num_generated_images).to(device)
    
    # Generate fake images
    with torch.no_grad():
        z = torch.randn(num_generated_images, latent_dim, device=device)
        fake_images = generator(z)
    
    # Convert generated images to uint8
    fake_images = ((fake_images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    
    # Compute FID
    fid = FrechetInceptionDistance(feature=2048)
    fid.to(device)
    
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    
    return fid.compute().item()

num_imgs = 2000

generator1 = Generator(100, 64, 3)
generator1.load_state_dict(torch.load('trained_models/generator_DCGAN.pth', map_location=torch.device('cpu')))
fid_score1 = compute_fid(generator1, 'img_align_celeba', num_generated_images=num_imgs)

generator2 = Generator(100, 64, 3)
generator2.load_state_dict(torch.load('trained_models/generator_WGANGP.pth', map_location=torch.device('cpu')))
fid_score2 = compute_fid(generator2, 'img_align_celeba', num_generated_images=num_imgs)

generator3 = Generator(100, 64, 3)
generator3.load_state_dict(torch.load('trained_models/generator_WGANGP_HQ.pth', map_location=torch.device('cpu')))
fid_score3 = compute_fid(generator3, 'celeba_hq_256', num_generated_images=num_imgs)


print(f"FID Score: {fid_score1}")
print(f"FID Score: {fid_score2}")
print(f"FID Score: {fid_score3}")