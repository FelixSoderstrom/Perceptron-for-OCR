import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.data_loader import get_dataset


def get_augmentation_transforms():
    """
    Get the transforms used for data augmentation.

    Returns:
        transforms.Compose: A composition of image transforms for augmentation
    """
    return transforms.Compose(
        [
            # These transforms expect tensor input (after ToTensor has been applied)
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
            ),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Normalize at the end
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_validation_transform():
    """
    Get the transforms used for validation data.

    Returns:
        transforms.Normalize: Normalization transform for validation data
    """
    return transforms.Normalize((0.5,), (0.5,))


def visualize_mnist_samples(num_samples=10):
    """
    Display a grid of random MNIST digits.

    Args:
        num_samples: Number of samples to display
    """
    # Load the MNIST dataset
    dataset = get_dataset(train=True, transform=transforms.ToTensor())

    # Set up the figure
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 2))

    # Get random indexes
    indices = torch.randperm(len(dataset))[:num_samples]

    # Display each image
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        axes[i].imshow(img.squeeze().numpy(), cmap="gray")
        axes[i].set_title(f"Digit: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    return plt


def visualize_augmentations(digit=None, num_augmentations=5):
    """
    Visualize the effect of data augmentation on MNIST digits.

    Args:
        digit: Specific digit to visualize, or None for random
        num_augmentations: Number of augmented versions to display

    Returns:
        str: Message indicating which digit was augmented
        plt: Matplotlib figure object
    """
    # Load the dataset
    dataset = get_dataset(train=True, transform=transforms.ToTensor())

    # Find a specific digit if requested
    if digit is not None:
        indices = [
            i for i, (_, label) in enumerate(dataset) if label == digit
        ]
        if not indices:
            raise ValueError(f"No images found with label {digit}")
        idx = np.random.choice(indices)
    else:
        idx = np.random.randint(0, len(dataset))

    original_img, label = dataset[idx]

    # Define the augmentation transforms (same as in our training pipeline)
    augmentation = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
            ),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )

    # Set up the figure
    fig, axes = plt.subplots(
        1, num_augmentations + 1, figsize=((num_augmentations + 1) * 1.5, 2)
    )

    # Display original image
    axes[0].imshow(original_img.squeeze().numpy(), cmap="gray")
    axes[0].set_title(f"Original: {label}")
    axes[0].axis("off")

    # Generate and display augmented images
    for i in range(num_augmentations):
        # Apply augmentations
        augmented_img = augmentation(original_img)

        # Display augmented image
        axes[i + 1].imshow(augmented_img.squeeze().numpy(), cmap="gray")
        axes[i + 1].set_title(f"Augmented {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    return plt
