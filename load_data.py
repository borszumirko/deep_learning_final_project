import tensorflow as tf
import os
import matplotlib.pyplot as plt

def load_celebA_images(directory):
    image_paths = []
    for img_name in os.listdir(directory):
        image_paths.append(os.path.join(directory, img_name))
    
    # Convert to a Dataset
    image_paths = tf.data.Dataset.from_tensor_slices(image_paths)

    # Load and preprocess images
    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # Decode as RGB image
        img = tf.image.resize(img, [128, 128])  # Resize image
        img = tf.cast(img, tf.float32)
        img = img / 255.0  # Normalize to the range [0, 1]
        return img

    images = image_paths.map(_load_image)

    return images

dataset_dir = 'img_align_celeba'
train_dataset = load_celebA_images(dataset_dir)

def plot_images(dataset):
    plt.figure(figsize=(10, 10))
    for idx, image in enumerate(dataset.take(25)):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

plot_images(train_dataset)
