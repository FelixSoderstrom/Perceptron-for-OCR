import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import numpy as np

from src.config.hyperparameters import hyperparameters as hp


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


def get_dataset(train: bool = True, transform=None):
    return datasets.MNIST(
        root="./src/data/",
        train=train,
        download=True,
        transform=transform,
    )


def get_dataloaders(
    batch_size=None,
    val_batch_size=None,
    val_split=None,
    custom_train_transform=None,
):
    """
    Gets and splits the MNIST training data into training and validation DataLoaders.

    Args:
        batch_size: Batch size for training (defaults to hyperparameter config)
        val_batch_size: Batch size for validation (defaults to hyperparameter config)
        val_split: Portion of training data to use for validation (defaults to hyperparameter config)
        custom_train_transform: Optional custom transform to apply to training data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Use hyperparameters from config if not explicitly provided
    batch_size = batch_size if batch_size is not None else hp["batch_size"]
    val_batch_size = (
        val_batch_size if val_batch_size is not None else hp["val_batch_size"]
    )
    val_split = val_split if val_split is not None else hp["val_split"]

    # Load the full training dataset with just the base transform
    full_dataset = get_dataset(transform=transforms.ToTensor())

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split the dataset
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Define the augmentation transforms for training data
    if custom_train_transform is not None:
        train_transform = custom_train_transform
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(
        f"Dataset split: {train_size} training samples (with augmentation), "
        f"{val_size} validation samples (no augmentation)."
    )
    print(f"Batch sizes: training={batch_size}, validation={val_batch_size}")

    return train_loader, val_loader


def load_test_dataset(batch_size=100):
    """
    Load the MNIST test dataset using torchvision.

    Args:
        batch_size: The batch size for the DataLoader

    Returns:
        DataLoader: A DataLoader for the MNIST test dataset
    """
    # Define normalization transform to match our training (-0.5, 0.5)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the test dataset
    test_dataset = TransformedSubset(
        get_dataset(train=False, transform=transform)
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Test dataset: {len(test_dataset)} samples")
    return test_loader


def get_digit_image(n):
    """
    Get a test image of a specific digit.

    Args:
        n: The digit (0-9) to retrieve

    Returns:
        Tuple of (tensor_image, original_image):
            - tensor_image: Normalized PyTorch tensor for model input
            - original_image: Raw image data for display
    """
    # Load test dataset without normalization for display
    display_transform = transforms.ToTensor()
    model_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset_display = get_dataset(train=False, transform=display_transform)
    dataset_model = get_dataset(train=False, transform=model_transform)

    # Find images of the requested digit
    # digit_indices = [
    #     i for i, (_, label) in enumerate(dataset_display) if label == n
    # ]

    digit_indices = []
    for i, (_, label) in enumerate(dataset_display):
        if label == n:
            digit_indices.append(i)

    if not digit_indices:
        raise ValueError(f"No images found with label {n}")

    # Choose a random image of the digit
    idx = torch.randint(0, len(digit_indices), (1,)).item()
    idx = digit_indices[idx]

    # Get both the display and model versions
    display_image, _ = dataset_display[idx]
    model_image, _ = dataset_model[idx]

    return model_image, display_image


def get_digit_examples(digit, n_examples=5, dataset=None):
    """
    Get examples of a specific digit from the test dataset.

    Args:
        digit: Target digit (0-9)
        n_examples: Number of examples to retrieve
        dataset: Dataset to use, defaults to test dataset

    Returns:
        List of tuples (image_tensor, index)
    """
    if dataset is None:
        dataset = get_dataset(train=False, transform=transforms.ToTensor())

    # Find all examples of the specified digit
    digit_indices = [
        i for i, (_, label) in enumerate(dataset) if label == digit
    ]

    # Randomly select n_examples
    selected_indices = np.random.choice(
        digit_indices, min(n_examples, len(digit_indices)), replace=False
    )

    # Get the images
    examples = [(dataset[idx][0], idx) for idx in selected_indices]

    return examples
