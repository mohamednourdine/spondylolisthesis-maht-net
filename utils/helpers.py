def load_image(image_path):
    """Load an image from the specified path."""
    from PIL import Image
    return Image.open(image_path)

def save_image(image, save_path):
    """Save an image to the specified path."""
    image.save(save_path)

def normalize_tensor(tensor, mean, std):
    """Normalize a tensor with the given mean and standard deviation."""
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize(tensor)

def denormalize_tensor(tensor, mean, std):
    """Denormalize a tensor with the given mean and standard deviation."""
    import torchvision.transforms as transforms
    denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                        std=[1/s for s in std])
    return denormalize(tensor)

def set_seed(seed):
    """Set the random seed for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)