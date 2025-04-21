import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from typing import Dict


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


def get_training_data():
    """Gets the 60000 training images for the PyTorch network"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data/MNIST",
        train=True,
        download=True,
        transform=transform,
    )

    # Divided into batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_loader
