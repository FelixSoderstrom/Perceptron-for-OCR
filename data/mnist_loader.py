import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import os
from typing import Dict, Tuple


def choose_number(
    pytorch: bool = False, numpy: bool = False
) -> Dict[str, int | np.ndarray | torch.Tensor]:
    """
    User chooses a number between 0 and 9.
    Returns the number and the vector in specified format.

    Args:
        pytorch: If True, the vector returned is a torch tensor.
        numpy: If True, the vector returned is a numpy array.

    Returns:
        A tuple containing the number and the image vector.
    """
    while True:
        try:
            n = int(input("Enter a number between 0-9: "))
            if n < 0 or n > 9:
                raise ValueError
            break
        except ValueError:
            print(
                "\033[91mInvalid input. Please enter a valid number.\033[0m"
            )
    return {"number": n, "vector": get_image(n, pytorch, numpy)}


def get_image(
    n: int, pytorch: bool = False, numpy: bool = False
) -> np.ndarray | torch.Tensor:
    """
    Get image data from a random test image where the label is equal to n.

    Args:
        n: The digit (0-9) to fetch an image for.
        pytorch: If True, the vector returned is a torch tensor.
        numpy: If True, the vector returned is a numpy array.

    Returns:
        A flattened and normalized vector of shape (784,) in numpy or torch format
    """
    # Define file paths
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/MNIST/MNIST/raw",
    )
    images_file = os.path.join(base_dir, "t10k-images-idx3-ubyte")
    labels_file = os.path.join(base_dir, "t10k-labels-idx1-ubyte")

    if not os.path.exists(images_file) or not os.path.exists(labels_file):
        print("Dataset not found. Downloading...")
        datasets.MNIST(
            root="./data/MNIST",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    test_images = read_mnist_images(images_file)
    test_labels = read_mnist_labels(labels_file)

    # Get a random image that matches n
    matching_indices = np.where(test_labels == n)[0]
    if len(matching_indices) == 0:
        raise ValueError(f"No images found with label {n}")
    random_idx = np.random.choice(matching_indices)
    image = test_images[random_idx].astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    # Convert to PyTorch if needed
    if pytorch:
        image = torch.from_numpy(image)

    return image


def read_mnist_images(filename):
    """Read MNIST images from IDX file format"""

    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError("Invalid magic number")

        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8)

        return images.reshape(num_images, rows * cols)


def read_mnist_labels(filename):
    """Read MNIST labels from IDX file format"""
    with open(filename, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError("Invalid magic number")
        num_items = int.from_bytes(f.read(4), "big")

        buffer = f.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


class TransformedSubset(Dataset):
    """Dataset wrapper that applies a transform to a subset of another dataset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders(
    batch_size=64,
    val_split=0.1667,  # ~10k out of 60k is about 16.67%
) -> Tuple[DataLoader, DataLoader]:
    """Gets and splits the MNIST training data into training and validation DataLoaders."""

    # Define the base transform that converts PIL images to tensors (no normalization yet)
    base_transform = transforms.ToTensor()

    # Load the full training dataset with just the base transform
    full_dataset = datasets.MNIST(
        root="./data/MNIST",
        train=True,
        download=True,
        transform=base_transform,
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split the dataset
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Define the augmentation transforms for training data
    train_transform = transforms.Compose(
        [
            # These transforms expect tensor input (since we already called ToTensor)
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
            ),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Normalize at the end
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Define transform for validation data (just normalization)
    val_transform = transforms.Normalize((0.5,), (0.5,))

    # Create wrapped datasets with appropriate transforms
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
        * 2,  # Often use larger batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(
        f"Dataset split: {train_size} training samples (with augmentation), "
        f"{val_size} validation samples (no augmentation)."
    )
    return train_loader, val_loader
